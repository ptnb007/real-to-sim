from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    image_path = Path("/mnt/c/Users/woom/Desktop/woom/mira_scene/input/lab_no_robot.jpg")
    output_path = Path("/home/woom/Mira-Scene/work/lab_no_robot/input/floor_mask.png")
    preview_path = Path("/mnt/c/Users/woom/Desktop/woom/mira_scene/results/lab_no_robot_depth/floor_mask.png")

    with Image.open(image_path) as source:
        width, height = source.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Open wood-floor patches selected from the reference photograph.
    draw.polygon(
        [
            (int(width * 0.32), int(height * 0.78)),
            (int(width * 0.53), int(height * 0.70)),
            (int(width * 0.71), height - 1),
            (int(width * 0.24), height - 1),
        ],
        fill=255,
    )
    draw.polygon(
        [
            (int(width * 0.78), int(height * 0.70)),
            (width - 1, int(height * 0.64)),
            (width - 1, height - 1),
            (int(width * 0.68), height - 1),
        ],
        fill=255,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    mask.save(preview_path)
    print(output_path)


if __name__ == "__main__":
    main()
