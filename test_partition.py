from utils.waterbirds import WaterbirdsDataset


def verify_partitions():
    train_dataset = WaterbirdsDataset(train=True)
    df = train_dataset.data
    partitions, partition_details = train_dataset.get_partitions(25)
    for i, part in enumerate(partitions):
        part_rows = df.loc[part]
        unique_places = part_rows["place"].unique()
        if len(unique_places) != 1:
            print(f"Partition {i} FAILED: contains multiple places: {unique_places}")
        else:
            print(f"Partition {i}: All samples from place {unique_places[0]}")
        y_counts = part_rows["y"].value_counts().to_dict()
        count_y0 = y_counts.get(0, 0)
        count_y1 = y_counts.get(1, 0)
        if count_y0 != count_y1:
            print(
                f"Partition {i} FAILED: Unbalanced labels: y=0 -> {count_y0} vs y=1 -> {count_y1}"
            )
        else:
            print(f"Partition {i}: Balanced labels: y=0 and y=1 count is {count_y0}")


if __name__ == "__main__":
    verify_partitions()
