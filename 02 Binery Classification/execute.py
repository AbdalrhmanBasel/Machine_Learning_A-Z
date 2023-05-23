from config.cfg import cfg
from dataset.dataset_split import split_data
from dataset.sportsmans_height import SportsmanHeight
from model.PlayersClassifier import PlayerClassifier
from plots.plot_precision_recall import plot_precision_recall


def main():
    # Load dataset
    data = SportsmanHeight()
    df = data()

    height = df['height']
    player_type = df['class']

    # Model.py
    model = PlayerClassifier(cfg=cfg)

    # Split dataset
    X_train, y_train, valid_X, valid_y, X_test, y_test = split_data(height, player_type,
                                                                    train_size=cfg.train_set_percent,
                                                                    valid_size=cfg.valid_set_percent,
                                                                    test_size=cfg.test_set_percent,
                                                                    random_state=42)

    # Calculate confidence
    confidences = model.calculate_confidences(X_test)
    # print(confidence)

    # Predict Players
    y_pred = model.classify(confidences)
    # print(y_pred)

    # evaluate the model on the testing set
    accuracy, precision, recall, f1_score = model.calculate_metrics(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)

    model.calculate_metrics(y_test, y_pred)

    # # TODO: Plot the precision-recall curve
    plot_precision_recall(y_test, y_pred)


if __name__ == '__main__':
    main()
