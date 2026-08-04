from camcanon3r.eth3d_archives import (
    build_eth3d_selection,
    parse_7z_slt,
    selected_paths,
)


def test_parse_7z_slt_files_and_directories() -> None:
    listing = """Path = archive.7z
Type = 7z

----------
Path = office\\images
Size = 0
Folder = +

Path = office\\images\\DSC_0001.JPG
Size = 123
Folder = -

Path = office\\cameras.txt
Size = 45
Folder = -
"""
    assert parse_7z_slt(listing) == {
        "office/images/DSC_0001.JPG": 123,
        "office/cameras.txt": 45,
    }


def test_build_eth3d_selection_freezes_first_sorted_views() -> None:
    scene = "office"
    names = ["DSC_0003.JPG", "DSC_0001.JPG", "DSC_0004.JPG", "DSC_0002.JPG"]
    raw = {
        f"{scene}/images/dslr_images/{name}": 100 + index
        for index, name in enumerate(names)
    }
    undistorted = {
        f"{scene}/images/dslr_images_undistorted/{name}": 200 + index
        for index, name in enumerate(names)
    }
    depth = {
        scene: {
            f"{scene}/ground_truth_depth/dslr_images/{name}": 300 + index
            for index, name in enumerate(names)
        }
    }
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        raw[f"{scene}/dslr_calibration_jpg/{filename}"] = 10
        undistorted[
            f"{scene}/dslr_calibration_undistorted/{filename}"
        ] = 20

    selection = build_eth3d_selection(
        scenes=[scene],
        undistorted_members=undistorted,
        raw_members=raw,
        depth_members=depth,
        views_per_scene=3,
    )
    selected_scene = selection["scenes"][0]
    assert selected_scene["image_names"] == [
        "DSC_0001.JPG",
        "DSC_0002.JPG",
        "DSC_0003.JPG",
    ]
    assert selected_paths(selection, "raw") == [
        f"{scene}/images/dslr_images/DSC_0001.JPG",
        f"{scene}/images/dslr_images/DSC_0002.JPG",
        f"{scene}/images/dslr_images/DSC_0003.JPG",
    ]
