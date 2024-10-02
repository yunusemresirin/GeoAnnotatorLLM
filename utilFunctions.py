import os

def clearGGUFDir(gguf_path: str, quantization: str) -> None:
    allowed_quantizations=[
        "not_quantized",
        "fast_quantized",
        "quantized",     
        "f32",     
        "f16",    
        "q8_0",  
        "q4_k_m",
        "q5_k_m",
        "q2_k",
        "q3_k_l",
        "q3_k_m",
        "q3_k_s",
        "q4_0",
        "q4_1",
        "q4_k_s",
        "q4_k",
        "q5_1",
        "q5_k_s",
        "q6_k",
        "iq2_xxs",
        "iq2_xs",
        "iq3_xxs",
        "q3_k_xs"
    ]
    if quantization not in allowed_quantizations:
        raise ValueError(f"Invalid quantizationparameter: {quantization}. Authorized values: {allowed_quantizations}")

    for filename in os.listdir(gguf_path):
        file_path = os.path.join(gguf_path, filename)
        if os.path.isfile(file_path) and filename != f"unsloth.{quantization.upper()}.gguf":
            try:
                os.remove(file_path)
                print(f"Removed file: {filename}")
            except Exception as e:
                print(f"Deletion of file {filename} failed. Cause: {e}")

    print("Deletion process completed!")