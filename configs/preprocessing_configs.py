CONFIGS = {

    # Config 1: matches the CT-RATE paper
    "config1": {
        "name": "spacing0.75-0.75-1.5_shape480-480-240_hu-1000-1000",
        "target_voxel_spacing": (0.75, 0.75, 1.5),
        "target_shape": (480, 480, 240),
        "hu_min": -1000,
        "hu_max": 1000,
        "bbox_margin": 30,
        "lung_only": False,
    },

    # Config 1 + hu_max:400
    "config1_hu400": {
        "name": "spacing0.75-0.75-1.5_shape480-480-240_hu-1000-400",
        "target_voxel_spacing": (0.75, 0.75, 1.5),
        "target_shape": (480, 480, 240),
        "hu_min": -1000,
        "hu_max": 400,
        "bbox_margin": 30,
        "lung_only": False,
    },


    # Config 1 + lung-only 
    "config1_lungonly": {
        "name": "spacing0.75-0.75-1.5_shape480-480-240_hu-1000-400_lungonly",
        "target_voxel_spacing": (0.75, 0.75, 1.5),
        "target_shape": (480, 480, 240),
        "hu_min": -1000,
        "hu_max": 400,
        "bbox_margin": 30,
        "lung_only": True,
    },

    # Config 2
    "config2": {
        "name": "spacing1.5-1.5-1.5_shape128-128-128_hu-1000-200",
        "target_voxel_spacing": (1.5, 1.5, 1.5),
        "target_shape": (128, 128, 128),
        "hu_min": -1000,
        "hu_max": 200,
        "bbox_margin": 30,
        "lung_only": False,
    },

}