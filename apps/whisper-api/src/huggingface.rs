use std::path::PathBuf;

use anyhow::Result;

pub fn create_hf_api() -> Result<hf_hub::api::sync::Api> {
  let mut builder = hf_hub::api::sync::ApiBuilder::new();

  let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_default();
  let cache_dir = std::env::var("HF_HOME").unwrap_or_default();
  let token = std::env::var("HF_TOKEN").unwrap_or_default();

  if !endpoint.is_empty() {
    builder = builder.with_endpoint(endpoint);
  }

  if !cache_dir.is_empty() {
    builder = builder.with_cache_dir(PathBuf::from(cache_dir));
  }

  if !token.is_empty() {
    builder = builder.with_token(Some(token));
  }

  Ok(builder.build()?)
}
