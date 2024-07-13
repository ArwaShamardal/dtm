import subprocess
import os
# dataset is in InstantMesh/data/objaverse/rendering_zero123plus
# format is 
# entry directory: indices, for example, 0, 1, 2 and so on
#  7 images in each folder each with the format of 3d.png, for example, 000.png, 001.png, where 000.png is the query image

def run_training(base_config, gpus, num_nodes):
    current_dir = os.getcwd()
    instant_mesh_dir = os.path.join(current_dir, "InstantMesh")

    print(os.path.basename(current_dir))
    if os.path.basename(current_dir) == "InstantMesh":
        print(f"Already in InstantMesh directory: {current_dir}")
        instant_mesh_dir = current_dir
    else:
        # Check if InstantMesh directory exists
        if os.path.exists(instant_mesh_dir):
            os.chdir(instant_mesh_dir)
            print(f"Changed to InstantMesh directory: {os.getcwd()}")
        else:
            print(f"InstantMesh directory not found at {instant_mesh_dir}")


    command = [
        'python', 'train.py',
        '--base', base_config,
        '--gpus', gpus,
        '--num_nodes', str(num_nodes)
    ]
    try:
        subprocess.run(command, check=True, shell= True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")

# Example usage
if __name__ == "__main__":
    base_config = 'configs/zero123plus-finetune.yaml'
    gpus = '0'
    num_nodes = 1
    run_training(base_config, gpus, num_nodes)
