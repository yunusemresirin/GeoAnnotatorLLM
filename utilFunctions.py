import os
import shutil

async def clearGGUFDir(gguf_path: str, quantization: str="q4_k_m") -> None:
    r"""
    Remove unused files when converting model to GGUF-file

    Args:
    - gguf_path (str): path to gguf-file of specific model
    - quantization (str): quantization method for converting model
    """

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

def clearMainDir() -> None:
    r"""
    Remove temporary folders by unsloth and checkpoints of the SFTTrainer
    """
    
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        
        if os.path.isdir(folder_path) and any(sub in folder_name for sub in ["unsloth", "outputs"]):
            shutil.rmtree(folder_path)
            print(f"Deleted: {folder_path}")