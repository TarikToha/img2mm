# radar parameters
range_bin_size = 42.0832 / 1000  # m
dop_bin_size = 42.4292 / 1000  # m/s

max_range_bins = 80
max_dop_bins = 32
max_azi_bins = 64
max_elev_bins = 32

min_range_dist = 1
dop_center = max_dop_bins // 2
max_azi_dist = 1.5

meta_labels = {
    'PositionCleanedV1Label.WITHOUT': 'no_weapon',
    'PositionCleanedV1Label.LEFT_ANKLE': 'left_ankle',
    'PositionCleanedV1Label.RIGHT_ANKLE': 'right_ankle',
    'PositionCleanedV1Label.LEFT_CHEST': 'left_chest',
    'PositionCleanedV1Label.RIGHT_CHEST': 'right_chest',
    'PositionCleanedV1Label.LEFT_POCKET': 'left_pocket',
    'PositionCleanedV1Label.RIGHT_POCKET': 'right_pocket',
    'PositionCleanedV1Label.LEFT_KNEE': 'left_knee',
    'PositionCleanedV1Label.RIGHT_KNEE': 'right_knee',
    'PositionCleanedV1Label.LEFT_WAIST': 'left_waist',
    'PositionCleanedV1Label.RIGHT_WAIST': 'right_waist',
}

object_mapping = {
    'data-mini_pc_std_obj_cascaded_dataset': ['std_brick_object', 'br'],
    'data-mini_pc_new_cascaded_dataset': ['std_t_object', 'nt'],
    'cascaded_dataset': ['old_t_object', 'ot']
}

weapon_labels = ['left_ankle', 'left_chest', 'left_pocket', 'left_waist', 'no_weapon',
                 'right_ankle', 'right_chest', 'right_pocket', 'right_waist']

visual_labels = {
    'clothes': ['normal', 'fleece_jacket', 'leather_jacket', 'snow_jacket'],
    'environments': ['normal', 'ladder', 'ladder_whiteboard', 'corridor'],
}

object_type = {
    'std_brick_object': 'br',
    'std_t_object': 'nt',
    'old_t_object': 'ot'
}

wall_labels = [
    'brick', 'brick_ladder', 'curtain',
    'furniture', 'furniture_ladder', 'gator', 'gator_ladder'
]


def generate_weapon_prompt(clothing_label: str, environment_label: str) -> str:
    clothing_descriptions = {
        "fleece_jacket": (
            "a fleece jacket",
            "creates slightly more diffuse radar reflections in the torso region, but has minimal impact on detection accuracy"
        ),
        "leather_jacket": (
            "a leather jacket",
            "introduces noticeable signal attenuation and scattering, potentially reducing detection reliability for concealed objects"
        ),
        "normal": (
            "normal clothing such as a shirt and pants",
            "produces clean, distinguishable radar reflections with minimal interference, forming the baseline for unarmed detection"
        ),
        "snow_jacket": (
            "a heavily padded snow jacket",
            "significantly attenuates and diffuses mmWave signals, often masking reflections from concealed objects"
        ),
    }

    environment_descriptions = {
        "corridor": (
            "walls introduce multipath reflections, causing ghosting artifacts in the torso and leg regions that may resemble concealed object signatures"
        ),
        "ladder": (
            "a ladder is present on the left side of the scene, introducing moderate environmental reflections that may interfere with clean spectrum interpretation"
        ),
        "ladder_whiteboard": (
            "both a ladder on the left and a whiteboard on the right produce multiple reflection paths, increasing the likelihood of clutter and potential false positives in the radar data"
        ),
        "normal": (
            "the environment is clean and minimally reflective, allowing for clear radar returns with minimal background interference"
        ),
    }

    PROMPT_TEMPLATE = (
        "A person is wearing {clothing_phrase}, which {clothing_effect}. "
        "The person is walking toward the radar in an environment where {environment_effect}. "
        "Generate the expected radar spectrum assuming the person is unarmed and no concealed objects are present."
    )

    clothing_phrase, clothing_effect = clothing_descriptions[clothing_label]
    environment_effect = environment_descriptions[environment_label]

    return PROMPT_TEMPLATE.format(
        clothing_phrase=clothing_phrase,
        clothing_effect=clothing_effect,
        environment_effect=environment_effect
    )


def generate_wall_prompt(wall_label: str) -> str:
    WALL_PROMPT_TEMPLATE = (
        "The radar is directed at a {wall_type}, which {wall_effect}. "
        "There is no human or object present behind the wall. "
        "Generate the expected radar spectrum for this blank scene."
    )

    wall_descriptions = {
        "brick": (
            "styrofoam wall",
            "permits most mmWave signals to pass through with low reflectivity, making it minimally obstructive to radar sensing"
        ),
        "brick_ladder": (
            "styrofoam wall with an additional reflective object such as a ladder",
            "permits most mmWave signals to pass through with low reflectivity, while the ladder introduces additional reflections in the scene"
        ),
        "curtain": (
            "curtain",
            "allows most mmWave signals to pass through with minimal reflection, regardless of its thickness"
        ),
        "furniture": (
            "particle board wall",
            "acts as a moderately reflective surface for mmWave signals, introducing structured reflections based on its material composition"
        ),
        "furniture_ladder": (
            "particle board wall with an additional reflective object such as a ladder",
            "acts as a moderately reflective surface, while the ladder introduces additional mmWave reflections into the scene"
        ),
        "gator": (
            "dense gator board wall",
            "introduces strong reflections"
        ),
        "gator_ladder": (
            "dense gator board wall with an additional reflective object such as a ladder",
            "introduces strong reflections along with clutter from the ladder"
        ),
    }

    wall_type, wall_effect = wall_descriptions[wall_label]

    return WALL_PROMPT_TEMPLATE.format(
        wall_type=wall_type,
        wall_effect=wall_effect
    )
