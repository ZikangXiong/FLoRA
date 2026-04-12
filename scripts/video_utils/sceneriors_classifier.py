import os
import json


def generate_and_classify_scenarios(root_dir):
    mapping = {}
    classification = {
        "Change Lane": [],
        "Following": [],
        "Near Static": [],
        "Near VRU": [],
        "Turn": [],
        "Stopping": [],
        "Starting": [],
        "Stationary": [],
        "Traversing": [],
    }

    for scenario_name in os.listdir(root_dir):
        scenario_path = os.path.join(root_dir, scenario_name)
        if os.path.isdir(scenario_path):
            videos = [
                f"/scenarios/videos/{scenario_name}/{file}"
                for file in os.listdir(scenario_path)
                if file.lower().endswith(".mp4")
            ]
            if videos:
                mapping[scenario_name] = videos

                # Classify the scenario
                if "changing_lane" in scenario_name:
                    classification["Change Lane"].extend(videos)
                elif "following" in scenario_name:
                    classification["Following"].extend(videos)
                elif any(
                    k in scenario_name
                    for k in ["near_barrier", "near_trafficcone", "near_construction"]
                ):
                    classification["Near Static"].extend(videos)
                elif any(
                    k in scenario_name
                    for k in [
                        "near_pedestrian",
                        "near_bike",
                        "behind_pedestrian",
                        "behind_bike",
                    ]
                ):
                    classification["Near VRU"].extend(videos)
                elif "turn" in scenario_name:
                    classification["Turn"].extend(videos)
                elif "stopping" in scenario_name:
                    classification["Stopping"].extend(videos)
                elif "starting" in scenario_name:
                    classification["Starting"].extend(videos)
                elif "stationary" in scenario_name:
                    classification["Stationary"].extend(videos)
                elif "traversing" in scenario_name:
                    classification["Traversing"].extend(videos)
                else:
                    print(f"Unclassified scenario: {scenario_name}")

    return mapping, classification


def main():
    root_dir = "docs/.vuepress/public/scenarios/videos"
    mapping_file = "docs/.vuepress/public/scenarios/videos/scenario_video_mapping.json"
    classification_file = (
        "docs/.vuepress/public/scenarios/videos/classified_scenarios.json"
    )

    mapping, classification = generate_and_classify_scenarios(root_dir)

    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)

    with open(classification_file, "w") as f:
        json.dump(classification, f, indent=2)

    print(f"Mapping saved to {mapping_file}")
    print(f"Classification saved to {classification_file}")


if __name__ == "__main__":
    main()
