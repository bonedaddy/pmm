# pmm

pmm (plex media manager) is a rust cli for managing plex libraries. It's primarily intended for creating catalogues of media contained in a library. Catalogues can be created in a few ways:

* Grouping together media under a directory (ie: `/media/sci_fi`)
* Grouping together media with similar names using levenshtein distance
* Grouping together media based on semantic analysis by using fasttext


# Semantic Analysis

For using the semantic analysis you need to download [this dataset](https://fasttext.cc/docs/en/crawl-vectors.html)