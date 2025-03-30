use fasttext::{Args, FastText};
use std::collections::{HashMap, HashSet};
use std::error::Error;

// FastText wrapper for semantic analysis
pub struct FastTextEmbedding {
    model: FastText,
    dimension: usize,
}

impl FastTextEmbedding {
    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut model = FastText::new();
        model.load_model(path).map_err(|e| e.to_string())?;
        let dimension = 300;

        log::info!("Loaded FastText model with dimension {}", dimension);
        Ok(Self { model, dimension })
    }

    pub fn get_sentence_embedding(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        self.model
            .get_sentence_vector(text)
            .map_err(|e| e.to_string().into())
    }

    pub fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let (mut dot_product, mut norm1, mut norm2) = (0.0, 0.0, 0.0);
        for (&x, &y) in vec1.iter().zip(vec2.iter()) {
            dot_product += x * y;
            norm1 += x * x;
            norm2 += y * y;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        dot_product / (norm1.sqrt() * norm2.sqrt())
    }

    pub fn is_similar_to_cached_terms(
        &self,
        directory: &str,
        term_embeddings: &HashMap<String, Vec<f32>>,
        threshold: f32,
    ) -> Result<bool, Box<dyn Error>> {
        let dir_embedding = self.get_sentence_embedding(directory)?;

        for (term, term_embedding) in term_embeddings {
            let similarity = self.cosine_similarity(&dir_embedding, term_embedding);

            log::debug!(
                "Similarity between '{directory}' and term '{term}': {:.4}",
                similarity
            );

            if similarity >= threshold {
                log::info!(
                    "Directory '{directory}' matches term '{term}' (similarity: {:.4})",
                    similarity
                );
                return Ok(true);
            }
        }

        Ok(false)
    }
}

// New struct to represent a directory with its embedding
#[derive(Clone)]
struct DirectoryEmbedding {
    name: String,
    path: String,
    embedding: Vec<f32>,
    item_ids: Vec<String>,
}

// New struct to represent a cluster of directories
#[derive(Clone)]
struct DirectoryCluster {
    centroid_idx: usize,  // Index of the directory closest to the centroid
    directories: Vec<String>,
    item_ids: HashSet<String>,
    keywords: Vec<String>, // Most representative terms
}
// Custom k-means implementation without external dependencies
fn kmeans_clustering(
    embeddings: &[Vec<f32>],
    k: usize,
    max_iterations: usize,
    similarity_threshold: f32,
) -> Vec<usize> {
    if embeddings.is_empty() || k == 0 || k > embeddings.len() {
        return vec![];
    }

    let n = embeddings.len();
    let dim = embeddings[0].len();
    
    // Initialize centroids by randomly selecting k data points
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut used = HashSet::new();
    
    while centroids.len() < k {
        let idx = fastrand::usize(0..n);
        if used.insert(idx) {
            centroids.push(embeddings[idx].clone());
        }
    }
    
    let mut assignments = vec![0; n];
    let mut last_assignments = vec![n; n]; // Initialize with invalid assignments
    
    for _ in 0..max_iterations {
        // Assign points to closest centroid if they meet the similarity threshold
        for i in 0..n {
            let mut closest_centroid = 0;
            let mut min_distance = f32::MAX;
            
            for (j, centroid) in centroids.iter().enumerate() {
                let mut distance = 0.0;
                for ((&e1, &e2)) in embeddings[i].iter().zip(centroid.iter()) {
                    let diff = e1 - e2;
                    distance += diff * diff;
                }
                
                if distance < min_distance {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }
            
            // Convert Euclidean distance to similarity (0 to 1 scale)
            // Higher distance = lower similarity
            let max_possible_distance = dim as f32 * 4.0; // Assuming normalized embeddings around 0
            let similarity = 1.0 - (min_distance.sqrt() / max_possible_distance.sqrt());
            
            // Only assign to cluster if similarity meets threshold
            if similarity >= similarity_threshold {
                assignments[i] = closest_centroid;
            } else {
                // Assign to "no cluster" (using n as an invalid index)
                assignments[i] = n; // n is out of bounds, marking as uncategorized
            }
        }
        
        // Check if assignments haven't changed
        if assignments == last_assignments {
            break;
        }
        
        last_assignments = assignments.clone();
        
        // Recompute centroids
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0; k];
        
        for i in 0..n {
            let cluster = assignments[i];
            counts[cluster] += 1;
            
            for j in 0..dim {
                new_centroids[cluster][j] += embeddings[i][j];
            }
        }
        
        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..dim {
                    new_centroids[i][j] /= counts[i] as f32;
                }
            } else {
                // If a centroid has no points, reinitialize it
                let random_idx = fastrand::usize(0..n);
                new_centroids[i] = embeddings[random_idx].clone();
            }
        }
        
        centroids = new_centroids;
    }
    
    assignments
}

// Function to extract keywords from a cluster
fn extract_keywords(
    directories: &[String],
    num_keywords: usize,
) -> Vec<String> {
    // Tokenize all directory names and count occurrences
    let mut token_count: HashMap<String, usize> = HashMap::new();
    
    for dir in directories {
        for word in dir.split(|c: char| !c.is_alphanumeric()) {
            let word = word.trim().to_lowercase();
            if word.len() > 2 { // Filter out very short words
                *token_count.entry(word).or_insert(0) += 1;
            }
        }
    }
    
    // Get the most frequent tokens
    let mut keywords: Vec<(String, usize)> = token_count.into_iter().collect();
    keywords.sort_by(|a, b| b.1.cmp(&a.1));
    
    keywords.iter()
        .take(num_keywords)
        .map(|(token, _)| token.clone())
        .collect()
}

pub async fn auto_categorize_by_clustering(
    manager: &mut crate::PlexCollectionManager,
    library_id: &str,
    base_path: &str,
    model_path: &str,
    num_clusters: usize,
    collection_prefix: &str,
    iterations: usize,
    similarity_threshold: f32, // New parameter to control minimum similarity
) -> Result<(), Box<dyn Error>> {
    let embedding_model = FastTextEmbedding::load(model_path)?;
    
    // Get all folders and their items
    let folder_items = manager.tag_items_by_folder(library_id).await?;
    let mut directory_embeddings: Vec<DirectoryEmbedding> = Vec::new();
    
    for (file_path, item_ids) in folder_items {
        let dir_name = manager.extract_directory(&file_path, base_path);
        
        match embedding_model.get_sentence_embedding(&dir_name) {
            Ok(embedding) => {
                directory_embeddings.push(DirectoryEmbedding {
                    name: dir_name,
                    path: file_path,
                    embedding,
                    item_ids: item_ids.into_iter().collect(),
                });
            }
            Err(e) => {
                log::warn!("Error computing embedding for directory '{}': {}", dir_name, e);
            }
        }
    }
    
    if directory_embeddings.is_empty() {
        return Err("No directories could be embedded".into());
    }
    
    log::info!("Generated embeddings for {} directories", directory_embeddings.len());
    
    // Extract embeddings for clustering
    let embeddings: Vec<Vec<f32>> = directory_embeddings
        .iter()
        .map(|dir| dir.embedding.clone())
        .collect();
    
    // Perform clustering with similarity threshold
    let assignments = kmeans_clustering(&embeddings, num_clusters, iterations, similarity_threshold);
    
    if assignments.is_empty() {
        return Err("Clustering failed".into());
    }
    
    // Count how many items were categorized vs. uncategorized
    let uncategorized_count = assignments.iter().filter(|&&a| a >= embeddings.len()).count();
    log::info!(
        "Categorized {} directories, {} directories below similarity threshold of {}",
        embeddings.len() - uncategorized_count,
        uncategorized_count,
        similarity_threshold
    );
    
    // Group directories by cluster
    let mut clusters = vec![
        DirectoryCluster {
            centroid_idx: 0,
            directories: Vec::new(),
            item_ids: HashSet::new(),
            keywords: Vec::new(),
        };
        num_clusters
    ];
    
    // Group directories by their assigned clusters (skip uncategorized)
    for (i, &cluster_idx) in assignments.iter().enumerate() {
        // Skip directories that didn't meet the similarity threshold
        if cluster_idx >= clusters.len() {
            log::debug!("Directory '{}' did not meet similarity threshold", directory_embeddings[i].name);
            continue;
        }
        
        let dir = &directory_embeddings[i];
        let cluster = &mut clusters[cluster_idx];
        
        if cluster.directories.is_empty() {
            cluster.centroid_idx = i; // First directory becomes the centroid
        }
        
        cluster.directories.push(dir.name.clone());
        for id in &dir.item_ids {
            cluster.item_ids.insert(id.clone());
        }
    }
    
    // Extract keywords for each cluster
    for cluster in &mut clusters {
        if !cluster.directories.is_empty() {
            cluster.keywords = extract_keywords(&cluster.directories, 5);
            
            // If no keywords extracted, use first directory as fallback
            if cluster.keywords.is_empty() {
                if let Some(first_dir) = cluster.directories.first() {
                    let parts: Vec<&str> = first_dir.split('/').collect();
                    if let Some(last_part) = parts.last() {
                        cluster.keywords = vec![last_part.to_string()];
                    }
                }
            }
        }
    }
    
    // Create collections for each valid cluster
    for (i, cluster) in clusters.iter().enumerate() {
        if cluster.item_ids.is_empty() {
            continue;
        }
        
        let keywords_str = if !cluster.keywords.is_empty() {
            cluster.keywords.join(" ")
        } else {
            format!("Cluster {}", i + 1)
        };
        
        let collection_name = format!("{} {}", collection_prefix, keywords_str);
        let item_list: Vec<String> = cluster.item_ids.iter().cloned().collect();
        
        log::info!(
            "Creating collection '{}' with {} items",
            collection_name,
            item_list.len()
        );
        
        // Check if collection already exists
        let existing_collections = manager.get_collections(library_id).await?;
        let existing = existing_collections.iter().find(|c| c.title == collection_name);
        
        match existing {
            Some(coll) => {
                log::info!(
                    "Collection '{}' already exists (ID: {}). Adding items...",
                    collection_name,
                    coll.rating_key
                );
                manager.add_items_to_collection(&coll.rating_key, &item_list).await?;
            }
            None => {
                let new_collection = manager.create_collection(library_id, &collection_name).await?;
                manager
                    .add_items_to_collection(&new_collection.rating_key, &item_list)
                    .await?;
                log::info!("Created and added items to new collection '{}'", collection_name);
            }
        }
    }
    
    Ok(())
}

pub async fn tag_by_fasttext(
    manager: &mut crate::PlexCollectionManager,
    library_id: &str,
    base_path: &str,
    terms: &[String],
    collection: &str,
    threshold: f64,
    model_path: &str,
) -> Result<(), Box<dyn Error>> {
    if terms.is_empty() {
        return Err("No valid terms provided for comparison".into());
    }

    let embedding_model = FastTextEmbedding::load(model_path)?;

    log::info!("Computing embeddings for terms: {:?}", terms);
    let mut term_embeddings = HashMap::new();
    for term in terms {
        let embedding = embedding_model.get_sentence_embedding(term)?;
        term_embeddings.insert(term.clone(), embedding);
    }

    let folder_items = manager.tag_items_by_folder(library_id).await?;
    let mut matching_items: HashSet<String> = HashSet::new();

    for (file_path, item_ids) in folder_items {
        let dir_name = manager.extract_directory(&file_path, base_path);

        match embedding_model.is_similar_to_cached_terms(&dir_name, &term_embeddings, threshold as f32) {
            Ok(true) => {
                matching_items.extend(item_ids);
            }
            Ok(false) => {}
            Err(e) => {
                log::warn!("Error comparing directory '{}': {}", dir_name, e);
            }
        }
    }

    if matching_items.is_empty() {
        log::info!("No matching items found for collection.");
        return Ok(());
    }

    log::info!("Found {} matching items.", matching_items.len());

    let existing_collections = manager.get_collections(library_id).await?;
    let existing = existing_collections.iter().find(|c| c.title == collection);

    let item_list: Vec<String> = matching_items.into_iter().collect();

    match existing {
        Some(coll) => {
            log::info!(
                "Collection '{}' already exists (ID: {}). Adding items...",
                collection,
                coll.rating_key
            );
            manager.add_items_to_collection(&coll.rating_key, &item_list).await?;
        }
        None => {
            let new_collection = manager.create_collection(library_id, collection).await?;
            manager
                .add_items_to_collection(&new_collection.rating_key, &item_list)
                .await?;
            log::info!("Created and added items to new collection '{}'", collection);
        }
    }

    Ok(())
}
