import os

def get_dataset_paths(name, root):

    """
        Return list of folder paths containing GOOD (normal) training images.
        Each dataset has its own directory structure.
    """

    name = name.lower()

    #golden_pcb board
    if name in ["golden_pcb_metafloor", "normal"]:
        return [root]

    # MVTec(15 categories)
    if name in ["mvtec", "mvtecad"]:
        subdirs = [
            os.path.join(root, d, "train", "good")
            for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d, "train", "good"))
        ]
        return subdirs

    # MVTec3D — similar, but uses 'rgb' folder under train
    elif name in ["mvtec3d"]:
        subdirs = [
            os.path.join(root, d, "train", "good", "rgb")
            for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d, "train", "good", "rgb"))
        ]
        return subdirs


    # Visa(10 categories)
    elif name in ["visa"]:
        subdirs = [
            os.path.join(root, d, "Data", "Images", "Normal")
            for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d, "Data", "Images", "Normal"))
        ]
        return subdirs
    
    elif name in ["mpdd"]:
        subdirs = [
            os.path.join(root, d, "train", "good")
            for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d, "train", "good"))
        ]
        return subdirs
    

    elif name in ["btad"]:
        subdirs = [
            os.path.join(root, d, "train", "ok")
            for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d, "train", "ok"))
        ]
        return subdirs


    #realiad(30 categories)
    elif name in ["realiad", "real-iad"]:
        subdirs = []
        for d in os.listdir(root):
            ok_dir = os.path.join(root, d, "OK")
            if not os.path.isdir(ok_dir):
                continue
            #Add all session subfolders (S001, S002, etc.)
            for s in os.listdir(ok_dir):
                s_path = os.path.join(ok_dir, s)
                if os.path.isdir(s_path):
                    subdirs.append(s_path)
        return subdirs

    raise ValueError(f"Unknown dataset: {name}")
