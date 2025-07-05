use std::path::PathBuf;

use anyhow::Result;

pub fn create_hf_api() -> Result<hf_hub::api::sync::Api> {
  let mut builder = hf_hub::api::sync::ApiBuilder::new();

  // Read HF_MIRROR environment variable for endpoint
  if let Ok(mirror_url) = std::env::var("HF_MIRROR") {
    builder = builder.with_endpoint(mirror_url);
  }

  // Read HF_HOME environment variable for cache directory
  if let Ok(cache_dir) = std::env::var("HF_HOME") {
    builder = builder.with_cache_dir(PathBuf::from(cache_dir));
  }

  Ok(builder.build()?)
}
