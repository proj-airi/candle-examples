// https://github.com/lucasjinreal/Kokoros/blob/4c18036a174d28ddec3181c99395af332c32b28d/kokoros/src/onnx/base.rs

#[cfg(feature = "metal")]
use ort::execution_providers::coreml::CoreMLExecutionProvider;
#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
use ort::{
  execution_providers::cpu::CPUExecutionProvider,
  session::{Session, builder::SessionBuilder},
};

pub trait OrtBase {
  fn load_model(
    &mut self,
    model_path: String,
  ) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    let providers = [CUDAExecutionProvider::default().build()];

    #[cfg(feature = "metal")]
    let providers = [CoreMLExecutionProvider::default().build(), CPUExecutionProvider::default().build()];

    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    let providers = [CPUExecutionProvider::default().build()];

    match SessionBuilder::new() {
      Ok(builder) => {
        let session = builder
          .with_execution_providers(providers)
          .map_err(|e| format!("Failed to build session: {e}"))?
          .commit_from_file(model_path)
          .map_err(|e| format!("Failed to commit from file: {e}"))?;
        self.set_sess(session);
        Ok(())
      },
      Err(e) => Err(format!("Failed to create session builder: {e}")),
    }
  }

  fn print_info(&self) {
    if let Some(session) = self.sess() {
      eprintln!("Input names:");
      for input in &session.inputs {
        eprintln!("  - {}", input.name);
      }
      eprintln!("Output names:");
      for output in &session.outputs {
        eprintln!("  - {}", output.name);
      }

      #[cfg(feature = "cuda")]
      eprintln!("Configured with: CUDA execution provider");

      #[cfg(feature = "metal")]
      eprintln!("Configured with: CoreML execution provider");

      #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
      eprintln!("Configured with: CPU execution provider");
    } else {
      eprintln!("Session is not initialized.");
    }
  }

  fn set_sess(
    &mut self,
    sess: Session,
  );
  fn sess(&self) -> Option<&Session>;
}
