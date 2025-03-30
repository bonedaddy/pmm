use clap::{Parser, Subcommand};
use regex::Regex;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    str::FromStr,
};
use strsim::normalized_levenshtein;
use tokio;
use tracing::level_filters::LevelFilter;

pub mod embedding_model;

// Define structures for Plex API responses
#[derive(Debug, Deserialize)]
struct MediaContainer<T>
where
    T: Default,
{
    #[serde(rename = "MediaContainer")]
    media_container: MediaContainerInner<T>,
}

#[derive(Debug, Deserialize)]
struct MediaContainerInner<T>
where
    T: Default,
{
    #[serde(default)]
    #[serde(rename = "Directory")]
    directory: Vec<T>,

    #[serde(default)]
    #[serde(rename = "Metadata")]
    metadata: Vec<T>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PlexLibrary {
    key: String,
    title: String,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PlexItem {
    #[serde(rename = "ratingKey")]
    rating_key: String,
    title: String,
    #[serde(default)]
    #[serde(rename = "Media")]
    media: Vec<PlexMedia>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PlexCollection {
    #[serde(rename = "ratingKey")]
    rating_key: String,
    title: String,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PlexMedia {
    #[serde(default)]
    #[serde(rename = "Part")]
    parts: Vec<PlexPart>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PlexPart {
    #[serde(default)]
    file: String,
}

// Plex Collection Manager
struct PlexCollectionManager {
    client: Client,
    base_url: String,
    token: String,
    machine_id: Option<String>,
}

impl PlexCollectionManager {
    fn new(base_url: &str, token: &str) -> Self {
        let mut client_builder = Client::builder().timeout(std::time::Duration::from_secs(30));

        let client = client_builder.build().unwrap_or_else(|_| Client::new());

        PlexCollectionManager {
            client,
            base_url: base_url.to_string(),
            token: token.to_string(),
            machine_id: None,
        }
    }

    // Get the Plex server's machine ID (needed for some operations)
    async fn get_machine_id(&mut self) -> Result<String, Box<dyn Error>> {
        if let Some(ref id) = self.machine_id {
            return Ok(id.clone());
        }

        let url = format!("{}/", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            // Store status information before consuming
            let status_code = response.status().as_u16();
            let status_reason = response
                .status()
                .canonical_reason()
                .unwrap_or("Unknown Error");
            let error_body = response.text().await?;

            return Err(format!(
                "Failed to get server info: {} {}: {}",
                status_code, status_reason, error_body
            )
            .into());
        }

        #[derive(Debug, Deserialize)]
        struct ServerInfo {
            #[serde(rename = "MediaContainer")]
            media_container: ServerInfoContainer,
        }

        #[derive(Debug, Deserialize)]
        struct ServerInfoContainer {
            #[serde(rename = "machineIdentifier")]
            machine_identifier: String,
        }

        let server_info: ServerInfo = response.json().await?;
        let machine_id = server_info.media_container.machine_identifier;

        // Store for future use
        self.machine_id = Some(machine_id.clone());

        Ok(machine_id)
    }

    async fn get_libraries(&self) -> Result<Vec<PlexLibrary>, Box<dyn Error>> {
        let url = format!("{}/library/sections", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(format!("Failed to get libraries: {}", response.status()).into());
        }

        let data: MediaContainer<PlexLibrary> = response.json().await?;
        Ok(data.media_container.directory)
    }

    async fn get_items_in_library(
        &self,
        library_id: &str,
    ) -> Result<Vec<PlexItem>, Box<dyn Error>> {
        let url = format!("{}/library/sections/{}/all", self.base_url, library_id);

        let response = self
            .client
            .get(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(format!("Failed to get items: {}", response.status()).into());
        }

        let data: MediaContainer<PlexItem> = response.json().await?;
        Ok(data.media_container.metadata)
    }

    async fn get_collections(
        &self,
        library_id: &str,
    ) -> Result<Vec<PlexCollection>, Box<dyn Error>> {
        let url = format!(
            "{}/library/sections/{}/collections",
            self.base_url, library_id
        );

        let response = self
            .client
            .get(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(format!("Failed to get collections: {}", response.status()).into());
        }

        let data: MediaContainer<PlexCollection> = response.json().await?;
        Ok(data.media_container.metadata)
    }

    async fn create_collection(
        &self,
        library_id: &str,
        collection_title: &str,
    ) -> Result<PlexCollection, Box<dyn Error>> {
        // Debug info
        log::info!(
            "Creating collection '{}' in library {}",
            collection_title,
            library_id
        );

        // First check if collection already exists
        let existing_collections = self.get_collections(library_id).await?;
        for collection in existing_collections {
            if collection.title == collection_title {
                log::info!(
                    "Collection '{}' already exists with ID: {}",
                    collection_title,
                    collection.rating_key
                );
                return Ok(collection);
            }
        }

        // Use the endpoint and parameters that we know work for this Plex server
        let url = format!("{}/library/collections", self.base_url);

        let params = [
            ("title", collection_title),
            ("smart", "0"),
            ("sectionId", library_id),
        ];

        log::info!(
            "Creating collection with URL: {} and params: {:?}",
            url,
            params
        );

        let response = self
            .client
            .post(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .query(&params)
            .send()
            .await?;

        if response.status() != StatusCode::OK && response.status() != StatusCode::CREATED {
            // Store status information before consuming the response
            let status_code = response.status().as_u16();
            let status_reason = response
                .status()
                .canonical_reason()
                .unwrap_or("Unknown Error");

            // Now we can safely consume the response body
            let error_body = response.text().await?;

            return Err(format!(
                "Failed to create collection: {} {}: {}",
                status_code, status_reason, error_body
            )
            .into());
        }

        // Try to parse the response
        match response.json::<MediaContainer<PlexCollection>>().await {
            Ok(data) => {
                if data.media_container.metadata.is_empty() {
                    // If the metadata is empty but status was success, the collection may have been created
                    log::info!("No metadata in response, searching for newly created collection");
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                    let collections = self.get_collections(library_id).await?;
                    for collection in collections {
                        if collection.title == collection_title {
                            log::info!(
                                "Found collection '{}' with ID: {}",
                                collection_title,
                                collection.rating_key
                            );
                            return Ok(collection);
                        }
                    }

                    return Err("No collection was created (empty metadata)".into());
                }

                log::info!(
                    "Collection created: {} (ID: {})",
                    data.media_container.metadata[0].title,
                    data.media_container.metadata[0].rating_key
                );

                Ok(data.media_container.metadata[0].clone())
            }
            Err(e) => {
                // If parsing fails, try an alternative approach
                log::info!("Failed to parse collection creation response: {}", e);

                // Wait a moment and then fetch all collections to find the new one
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                let collections = self.get_collections(library_id).await?;
                for collection in collections {
                    if collection.title == collection_title {
                        log::info!(
                            "Found collection '{}' with ID: {}",
                            collection_title,
                            collection.rating_key
                        );
                        return Ok(collection);
                    }
                }

                Err("Failed to create or find collection".into())
            }
        }
    }

    async fn add_items_to_collection(
        &mut self,
        collection_id: &str,
        item_ids: &[String],
    ) -> Result<(), Box<dyn Error>> {
        log::info!(
            "Adding {} items to collection {}",
            item_ids.len(),
            collection_id
        );

        // Check if empty
        if item_ids.is_empty() {
            log::info!("No items to add");
            return Ok(());
        }

        // Get machine ID for server (required by Plex API)
        let machine_id = self.get_machine_id().await?;

        // Add items one at a time to ensure maximum compatibility
        let mut successful_additions = 0;
        let mut failed_additions = 0;

        for (index, item_id) in item_ids.iter().enumerate() {
            let url = format!(
                "{}/library/collections/{}/items",
                self.base_url, collection_id
            );

            // Format single item URI
            let uri_param = format!(
                "server://{}/com.plexapp.plugins.library/library/metadata/{}",
                machine_id, item_id
            );

            if index % 10 == 0 {
                log::info!("Processing item {} of {}", index + 1, item_ids.len());
            }

            // Try to add the item
            let response = match self
                .client
                .put(&url)
                .header("X-Plex-Token", &self.token)
                .query(&[("uri", &uri_param)])
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    log::error!("Network error adding item {}: {}", item_id, e);
                    failed_additions += 1;
                    continue;
                }
            };

            if response.status() != StatusCode::OK && response.status() != StatusCode::CREATED {
                let status = response.status().as_u16();
                failed_additions += 1;

                // Only log detailed error for the first few failures
                if failed_additions <= 3 {
                    match response.text().await {
                        Ok(body) => {
                            log::error!(
                                "Error adding item {}: Status {}: {}",
                                item_id,
                                status,
                                body
                            )
                        }
                        Err(_) => log::error!("Error adding item {}: Status {}", item_id, status),
                    }
                } else if failed_additions == 4 {
                    log::error!("Suppressing further error messages...");
                }
            } else {
                successful_additions += 1;
            }

            // Add a small delay between items to avoid overwhelming the server
            // Only sleep every few items to balance speed and server load
            if index % 5 == 4 {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
        }

        if failed_additions > 0 {
            log::error!(
                "Warning: Failed to add {} out of {} items to collection",
                failed_additions,
                item_ids.len()
            );
        }

        log::info!(
            "Successfully added {} items to collection",
            successful_additions
        );

        // If we managed to add at least one item, consider it a success
        if successful_additions > 0 {
            Ok(())
        } else {
            Err(format!("Failed to add any items to collection {}", collection_id).into())
        }
    }
    async fn remove_items_from_collection(
        &self,
        collection_id: &str,
        item_ids: &[String],
    ) -> Result<(), Box<dyn Error>> {
        for item_id in item_ids {
            let url = format!(
                "{}/library/collections/{}/items/{}",
                self.base_url, collection_id, item_id
            );

            let response = self
                .client
                .delete(&url)
                .header("X-Plex-Token", &self.token)
                .send()
                .await?;

            if response.status() != StatusCode::OK && response.status() != StatusCode::NO_CONTENT {
                return Err(format!(
                    "Failed to remove item {} from collection: {}",
                    item_id,
                    response.status()
                )
                .into());
            }
        }

        Ok(())
    }

    async fn delete_collection(&self, collection_id: &str) -> Result<(), Box<dyn Error>> {
        let url = format!("{}/library/collections/{}", self.base_url, collection_id);

        let response = self
            .client
            .delete(&url)
            .header("X-Plex-Token", &self.token)
            .send()
            .await?;

        if response.status() != StatusCode::OK && response.status() != StatusCode::NO_CONTENT {
            return Err(format!("Failed to delete collection: {}", response.status()).into());
        }

        Ok(())
    }

    async fn find_items_by_title(
        &self,
        library_id: &str,
        title_query: &str,
    ) -> Result<Vec<PlexItem>, Box<dyn Error>> {
        let url = format!("{}/library/sections/{}/all", self.base_url, library_id);

        let response = self
            .client
            .get(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .query(&[("title", title_query)])
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(format!("Failed to find items: {}", response.status()).into());
        }

        let data: MediaContainer<PlexItem> = response.json().await?;
        Ok(data.media_container.metadata)
    }

    async fn get_all_items(&self, library_id: &str) -> Result<Vec<PlexItem>, Box<dyn Error>> {
        let url = format!("{}/library/sections/{}/all", self.base_url, library_id);

        let response = self
            .client
            .get(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(format!("Failed to get items: {}", response.status()).into());
        }

        let data: MediaContainer<PlexItem> = response.json().await?;
        Ok(data.media_container.metadata)
    }

    // Get all unique folder paths from a library
    async fn get_folders(&self, library_id: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let items = self.get_all_items(library_id).await?;

        let mut folders = std::collections::HashSet::new();

        for item in items {
            for media in &item.media {
                for part in &media.parts {
                    // Extract directory path by finding the last slash
                    if let Some(last_slash_pos) = part.file.rfind('/') {
                        // Only include up to the last slash to get the directory
                        let directory = &part.file[0..last_slash_pos];
                        folders.insert(directory.to_string());
                    } else if let Some(last_slash_pos) = part.file.rfind('\\') {
                        // Handle Windows-style paths too
                        let directory = &part.file[0..last_slash_pos];
                        folders.insert(directory.to_string());
                    }
                }
            }
        }

        let mut folder_vec: Vec<String> = folders.into_iter().collect();
        folder_vec.sort();

        Ok(folder_vec)
    }

    // Tag items based on their folder path
    async fn tag_items_by_folder(
        &mut self,
        library_id: &str,
    ) -> Result<Vec<(String, Vec<String>)>, Box<dyn Error>> {
        let items = self.get_all_items(library_id).await?;

        let mut folder_items: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for item in items {
            if item.media.is_empty() || item.media[0].parts.is_empty() {
                continue;
            }

            // Get the file path from the first media part
            let file_path = &item.media[0].parts[0].file;

            // Store the full path and the item ID
            folder_items
                .entry(file_path.clone())
                .or_insert_with(Vec::new)
                .push(item.rating_key.clone());
        }

        // Convert HashMap to Vec for easier sorting and return
        let mut result: Vec<(String, Vec<String>)> = folder_items.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(result)
    }
    async fn get_collection_items(
        &self,
        collection_id: &str,
    ) -> Result<Vec<PlexItem>, Box<dyn Error>> {
        let url = format!(
            "{}/library/collections/{}/children",
            self.base_url, collection_id
        );

        let response = self
            .client
            .get(&url)
            .header("X-Plex-Token", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(format!("Failed to get collection items: {}", response.status()).into());
        }

        let data: MediaContainer<PlexItem> = response.json().await?;
        Ok(data.media_container.metadata)
    }
    // Extract directory name from a full path
    fn extract_directory(&mut self, file_path: &str, base_path: &str) -> String {
        // Strip base path
        let relative_path = if file_path.starts_with(base_path) {
            file_path[base_path.len()..]
                .trim_start_matches(&['/', '\\'][..])
                .to_string()
        } else {
            file_path.to_string()
        };

        // Extract directory
        if let Some(last_slash) = relative_path.rfind('/') {
            relative_path[..last_slash].to_string()
        } else if let Some(last_slash) = relative_path.rfind('\\') {
            relative_path[..last_slash].to_string()
        } else {
            relative_path
        }
    }

    // Find directories similar to a reference string
    async fn find_similar_directories(
        &mut self,
        library_id: &str,
        base_path: &str,
        compare_string: &str,
        threshold: f64,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        // Get folder-to-items mapping
        let folder_items = self.tag_items_by_folder(library_id).await?;

        // Find directories similar to the reference string
        let mut matching_items = Vec::new();

        for (file_path, item_ids) in folder_items {
            let dir_name = self.extract_directory(&file_path, base_path);

            // Compare directory to reference string
            let similarity = normalized_levenshtein(&dir_name, compare_string);

            if similarity >= threshold {
                log::info!(
                    "Directory '{}' matches reference '{}' with similarity {:.2}",
                    dir_name,
                    compare_string,
                    similarity
                );
                matching_items.extend(item_ids);
            }
        }

        // Remove duplicates
        matching_items.sort();
        matching_items.dedup();

        Ok(matching_items)
    }
}

// CLI interface using clap
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Plex server URL (e.g., http://localhost:32400)
    #[arg(short, long)]
    server: String,

    /// Plex authentication token
    #[arg(short, long)]
    token: String,

    #[command(subcommand)]
    command: Commands,

    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// List all libraries
    ListLibraries {},

    /// List all collections in a library
    ListCollections {
        /// Library ID
        #[arg(short, long)]
        library_id: String,
    },

    /// List all folders in a library
    ListFolders {
        /// Library ID
        #[arg(short, long)]
        library_id: String,
    },

    /// Create a new collection
    CreateCollection {
        /// Library ID
        #[arg(short, long)]
        library_id: String,

        /// Collection title
        #[arg(short, long)]
        title: String,
    },

    /// Add items to a collection
    AddToCollection {
        /// Collection ID
        #[arg(short, long)]
        collection_id: String,

        /// Item IDs to add (comma-separated)
        #[arg(short, long)]
        item_ids: String,
    },

    /// Remove items from a collection
    RemoveFromCollection {
        /// Collection ID
        #[arg(short, long)]
        collection_id: String,

        /// Item IDs to remove (comma-separated)
        #[arg(short, long)]
        item_ids: String,
    },

    /// Delete a collection
    DeleteCollection {
        /// Collection ID
        #[arg(short, long)]
        collection_id: String,
    },

    /// Search for items by title
    FindItems {
        /// Library ID
        #[arg(short, long)]
        library_id: String,

        /// Title query
        #[arg(short, long)]
        query: String,
    },

    /// Create collection and add items in one command
    TagItems {
        /// Library ID
        #[arg(short, long)]
        library_id: String,

        /// Collection title
        #[arg(short, long)]
        collection: String,

        /// Title search query to find items
        #[arg(short, long)]
        query: String,
    },

    /// Tag items based on their folder path
    #[command(about = "tag items in a collection based on the folder path")]
    TagByFolder {
        /// Library ID
        #[arg(short, long)]
        library_id: String,

        /// Only create collections for specific folders (comma-separated, optional)
        #[arg(short, long, help = "tag items which reside in this folder path")]
        folders: Option<String>,

        /// Prefix to add to collection names (optional)
        #[arg(short, long, help = "override the collection name items are tagged in")]
        prefix: Option<String>,
    },
    #[command(about = "tag items in a collection using pattern matching for the folder path")]
    /// Tag items based on regex pattern matching their file paths
    TagByRegex {
        /// Library ID
        #[arg(short, long)]
        library_id: String,

        /// Regex pattern to match against file paths
        #[arg(short, long)]
        pattern: String,

        /// Collection name to add matching items to
        #[arg(short, long)]
        collection: String,
    },
    /// List all items in a collection
    ListCollectionItems {
        /// Collection ID
        #[arg(short, long)]
        collection_id: String,
    },
    /// Tag items using word embeddings for semantic matching
    #[command(about = "tag items using word embeddings for semantic analysis")]
    TagByEmbeddings {
        /// Library ID
        #[arg(long)]
        library_id: String,

        /// Base path to strip from directories before semantic comparison
        #[arg(long)]
        base_path: String,

        /// Reference terms to compare directories against
        #[arg(long, value_delimiter = ',')]
        terms: Vec<String>,

        /// Collection name to use for matching items
        #[arg(long)]
        collection: String,

        /// Similarity threshold (0.0 to 1.0, where higher means more similar)
        #[arg(long, default_value = "0.6")]
        threshold: f64,

        /// Path to pre-trained word embedding model file
        #[arg(long)]
        model_path: String,
    },
    /// Tag items using word embeddings for semantic matching
    #[command(about = "tag items using word embeddings for semantic analysis")]
    TagClusteredByEmbeddings {
        /// Library ID
        #[arg(long)]
        library_id: String,

        /// Base path to strip from directories before semantic comparison
        #[arg(long)]
        base_path: String,

        /// Reference terms to compare directories against
        #[arg(long, value_delimiter = ',')]
        terms: Vec<String>,

        /// Collection name to use for matching items
        #[arg(long)]
        collection: String,

        /// Similarity threshold (0.0 to 1.0, where higher means more similar)
        #[arg(long, default_value = "0.6")]
        threshold: f64,

        /// Path to pre-trained word embedding model file
        #[arg(long)]
        model_path: String,

        #[arg(long)]
        cluster_size: usize,

        #[arg(long)]
        iterations: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::from_str(&cli.log_level)?)
        .init();

    let mut manager = PlexCollectionManager::new(&cli.server, &cli.token);

    match &cli.command {
        Commands::ListLibraries {} => {
            let libraries = manager.get_libraries().await?;
            log::info!("Libraries:");
            for lib in libraries {
                log::info!("  {} (ID: {})", lib.title, lib.key);
            }
        }

        Commands::ListCollections { library_id } => {
            let collections = manager.get_collections(library_id).await?;
            log::info!("Collections in library {}:", library_id);
            for collection in collections {
                log::info!("  {} (ID: {})", collection.title, collection.rating_key);
            }
        }

        Commands::ListFolders { library_id } => {
            let folders = manager.get_folders(library_id).await?;
            log::info!("Folders in library {}:", library_id);
            for folder in folders {
                log::info!("  {}", folder);
            }
        }

        Commands::CreateCollection { library_id, title } => {
            let collection = manager.create_collection(library_id, title).await?;
            log::info!(
                "Collection created: {} (ID: {})",
                collection.title,
                collection.rating_key
            );
        }

        Commands::AddToCollection {
            collection_id,
            item_ids,
        } => {
            let ids: Vec<String> = item_ids.split(',').map(String::from).collect();
            manager.add_items_to_collection(collection_id, &ids).await?;
            log::info!("Added {} items to collection {}", ids.len(), collection_id);
        }

        Commands::RemoveFromCollection {
            collection_id,
            item_ids,
        } => {
            let ids: Vec<String> = item_ids.split(',').map(String::from).collect();
            manager
                .remove_items_from_collection(collection_id, &ids)
                .await?;
            log::info!(
                "Removed {} items from collection {}",
                ids.len(),
                collection_id
            );
        }

        Commands::DeleteCollection { collection_id } => {
            manager.delete_collection(collection_id).await?;
            log::info!("Collection {} deleted", collection_id);
        }

        Commands::FindItems { library_id, query } => {
            let items = manager.find_items_by_title(library_id, query).await?;
            log::info!("Found {} items matching \"{}\":", items.len(), query);
            for item in items {
                log::info!("  {} (ID: {})", item.title, item.rating_key);
            }
        }

        Commands::TagItems {
            library_id,
            collection,
            query,
        } => {
            // Find items
            let items = manager.find_items_by_title(library_id, query).await?;
            log::info!("Found {} items matching \"{}\"", items.len(), query);

            if items.is_empty() {
                log::info!("No items found to add to collection.");
                return Ok(());
            }

            // Create collection
            let new_collection = manager.create_collection(library_id, collection).await?;
            log::info!(
                "Collection created: {} (ID: {})",
                new_collection.title,
                new_collection.rating_key
            );

            // Add items to collection
            let item_ids: Vec<String> = items.iter().map(|item| item.rating_key.clone()).collect();
            manager
                .add_items_to_collection(&new_collection.rating_key, &item_ids)
                .await?;
            log::info!("Added {} items to collection", item_ids.len());
        }
        // Here's the fixed part of the TagByFolder command implementation
        // Replace the entire TagByFolder match arm with this code:
        Commands::TagByFolder {
            library_id,
            folders,
            prefix,
        } => {
            // Get folder-to-items mapping
            let folder_items = manager.tag_items_by_folder(library_id).await?;

            log::info!("Found {} unique file paths in library", folder_items.len());

            // Filter folders if specified
            let selected_folders: Option<Vec<String>> = folders
                .as_ref()
                .map(|f| f.split(',').map(|s| s.trim().to_string()).collect());

            // Track created collections
            let mut created_collections = 0;
            let mut updated_collections = 0;
            let mut total_items_added = 0;

            // Debug output for folder paths
            if let Some(ref selected) = selected_folders {
                log::info!(
                    "Looking for paths containing any of these folders: {:?}",
                    selected
                );
            }

            // Get all existing collections to check for duplicates
            let existing_collections = match manager.get_collections(library_id).await {
                Ok(collections) => collections,
                Err(e) => {
                    log::error!("Failed to get existing collections: {}", e);
                    vec![]
                }
            };

            // Map to store collection names and the items we're adding to them
            let mut collection_items: std::collections::HashMap<String, Vec<String>> =
                std::collections::HashMap::new();

            // Process all files to determine which items go into which collections
            for (file_path, item_ids) in folder_items {
                // Skip empty items list
                if item_ids.is_empty() {
                    continue;
                }

                // Check if this file path contains any of the selected folders
                let should_include = if let Some(ref selected) = selected_folders {
                    selected.iter().any(|folder| file_path.contains(folder))
                } else {
                    true // Include all if no filter
                };

                if !should_include {
                    continue;
                }

                // Get folder name for collection - use the specified prefix or extract from path
                let collection_name = match prefix {
                    Some(ref p) => p.clone(),
                    None => {
                        // Extract folder name from path
                        if let Some(last_slash) = file_path.rfind('/') {
                            if let Some(second_last_slash) = file_path[..last_slash].rfind('/') {
                                file_path[second_last_slash + 1..last_slash].to_string()
                            } else {
                                file_path[..last_slash].to_string()
                            }
                        } else if let Some(last_slash) = file_path.rfind('\\') {
                            if let Some(second_last_slash) = file_path[..last_slash].rfind('\\') {
                                file_path[second_last_slash + 1..last_slash].to_string()
                            } else {
                                file_path[..last_slash].to_string()
                            }
                        } else {
                            "Unknown".to_string()
                        }
                    }
                };

                log::info!(
                    "Adding items from path: {} to collection \"{}\"",
                    file_path,
                    collection_name
                );

                // Add item IDs to the corresponding collection in our map
                collection_items
                    .entry(collection_name)
                    .or_insert_with(Vec::new)
                    .extend(item_ids.clone());
            }

            // Now process each collection and add items
            for (collection_name, item_ids) in collection_items {
                // Remove duplicates from item_ids
                let mut unique_item_ids = item_ids.clone();
                unique_item_ids.sort();
                unique_item_ids.dedup();

                log::info!(
                    "Processing collection \"{}\" with {} unique items",
                    collection_name,
                    unique_item_ids.len()
                );

                // Check if collection already exists
                let existing_collection = existing_collections
                    .iter()
                    .find(|c| c.title == collection_name);

                match existing_collection {
                    Some(collection) => {
                        log::info!(
                            "Collection \"{}\" already exists with ID: {}",
                            collection_name,
                            collection.rating_key
                        );

                        // Add items to existing collection
                        match manager
                            .add_items_to_collection(&collection.rating_key, &unique_item_ids)
                            .await
                        {
                            Ok(_) => {
                                log::info!(
                                    "Added {} items to existing collection \"{}\"",
                                    unique_item_ids.len(),
                                    collection_name
                                );
                                updated_collections += 1;
                                total_items_added += unique_item_ids.len();
                            }
                            Err(e) => {
                                log::error!(
                                    "Error adding items to existing collection \"{}\": {}",
                                    collection_name,
                                    e
                                );
                            }
                        }
                    }
                    None => {
                        // Create new collection
                        match manager
                            .create_collection(library_id, &collection_name)
                            .await
                        {
                            Ok(new_collection) => {
                                // Add items to collection
                                match manager
                                    .add_items_to_collection(
                                        &new_collection.rating_key,
                                        &unique_item_ids,
                                    )
                                    .await
                                {
                                    Ok(_) => {
                                        log::info!(
                                            "Added {} items to new collection \"{}\"",
                                            unique_item_ids.len(),
                                            collection_name
                                        );
                                        created_collections += 1;
                                        total_items_added += unique_item_ids.len();
                                    }
                                    Err(e) => {
                                        log::error!(
                                            "Error adding items to new collection \"{}\": {}",
                                            collection_name,
                                            e
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "Error creating collection \"{}\": {}",
                                    collection_name,
                                    e
                                );
                            }
                        }
                    }
                }
            }

            log::info!("Created {} new collections, updated {} existing collections, with {} total items added", 
             created_collections, updated_collections, total_items_added);
        }
        Commands::TagByRegex {
            library_id,
            pattern,
            collection,
        } => {
            // Create regex pattern
            let regex = match Regex::new(pattern) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Invalid regex pattern: {}", e);
                    return Err("Invalid regex pattern".into());
                }
            };

            log::info!("Looking for items with paths matching pattern: {}", pattern);

            // Get folder-to-items mapping
            let folder_items = manager.tag_items_by_folder(library_id).await?;
            log::info!("Found {} unique file paths in library", folder_items.len());

            // Collect all items that match the regex
            let mut matching_item_ids = Vec::new();

            for (file_path, item_ids) in &folder_items {
                if regex.is_match(file_path) {
                    log::info!("Matched path: {}", file_path);
                    matching_item_ids.extend(item_ids.clone());
                }
            }

            // Remove duplicates
            matching_item_ids.sort();
            matching_item_ids.dedup();

            log::info!(
                "Found {} unique items matching the pattern",
                matching_item_ids.len()
            );

            if matching_item_ids.is_empty() {
                log::info!("No items found to add to collection.");
                return Ok(());
            }

            // Get all existing collections to check for duplicates
            let existing_collections = manager.get_collections(library_id).await?;

            // Check if collection already exists
            let existing_collection = existing_collections.iter().find(|c| c.title == *collection);

            let collection_id = match existing_collection {
                Some(c) => {
                    log::info!(
                        "Collection \"{}\" already exists with ID: {}",
                        collection,
                        c.rating_key
                    );
                    c.rating_key.clone()
                }
                None => {
                    // Create new collection
                    let new_collection = manager.create_collection(library_id, collection).await?;
                    log::info!(
                        "Created new collection \"{}\" with ID: {}",
                        new_collection.title,
                        new_collection.rating_key
                    );
                    new_collection.rating_key
                }
            };

            // Add items to collection
            manager
                .add_items_to_collection(&collection_id, &matching_item_ids)
                .await?;
            log::info!(
                "Added {} items to collection \"{}\"",
                matching_item_ids.len(),
                collection
            );
        }
        Commands::ListCollectionItems { collection_id } => {
            let items = manager.get_collection_items(collection_id).await?;
            log::info!("Items in collection {}:", collection_id);
            if items.is_empty() {
                log::info!("  No items found in this collection");
            } else {
                log::info!("Total: {} items", items.len());
                for item in items {
                    log::info!("  {} (ID: {})", item.title, item.rating_key);
                }
            }
        }
        Commands::TagByEmbeddings {
            library_id,
            base_path,
            terms,
            collection,
            threshold,
            model_path,
        } => {
            // Call the word embedding implementation
            crate::embedding_model::tag_by_fasttext(
                &mut manager,
                library_id,
                base_path,
                terms,
                collection,
                *threshold,
                model_path,
            )
            .await?;
        }
        Commands::TagClusteredByEmbeddings {
            library_id,
            base_path,
            terms,
            collection,
            threshold,
            model_path,
            iterations,
            cluster_size
        } => {
            // Call the word embedding implementation
            crate::embedding_model::auto_categorize_by_clustering(
                &mut manager,
                &library_id,
                &base_path,
                &model_path,
                *cluster_size,
                &collection,
                *iterations,
                *threshold as f32,
            )
            .await?;
        }
    }

    Ok(())
}
