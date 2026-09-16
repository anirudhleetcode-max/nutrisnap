"""Food-recognition pipeline (runtime side).

    bytes --preprocess.decode_image--> RGB image (validated)
          --preprocess.clip_tensor----> 3x224x224 float32
          --clip_model.ClipEncoder----> 512-d image embedding (pretrained CLIP ViT-B/32, ONNX)
          --zero-shot (label matrix) / probe.LinearProbe--> class probabilities
          --personalize.knn_probs-----> per-user blend from past corrections
          --calibration.apply_temperature--> calibrated probabilities
          --pipeline.FoodRecognizer.decide--> "confident" | "uncertain" | "not_food"
"""
