import pandas as pd

# Ground Truth Daten
ground_truth_data = [
    {"ID": 0, "Text": "Unable to order after registration as the account was frozen and documents were requested The review took over a week with no resolution", "Similar_IDs": "0"},
    {"ID": 1, "Text": "Multiple orders had issues One was delivered late and the second was undelivered because the address was missing on the packaging", "Similar_IDs": "1"},
    {"ID": 2, "Text": "Delivery instructions were ignored The package was sent despite the account being closed and the delivery settings being missed", "Similar_IDs": "2"},
    {"ID": 3, "Text": "The advertised price increased during the purchase process Customer service was unresponsive after multiple attempts", "Similar_IDs": "3"},  # Komma hinzugefügt
    {"ID": 4, "Text": "Unauthorized charges for Prime membership occurred without an active subscription The credit card had to be canceled to stop the payments", "Similar_IDs": "4"},
    {"ID": 5, "Text": "Customer service was unprofessional with loud background noise and minimal help provided", "Similar_IDs": "5"},
    {"ID": 6, "Text": "A used item was sold as new and this issue happens frequently rather than being a one-time mistake", "Similar_IDs": "6"},
    {"ID": 7, "Text": "The experience with Prime was positive with convenient shipping and a wide product selection", "Similar_IDs": "7"},  # Komma hinzugefügt
    {"ID": 8, "Text": "The job application process was poorly managed A video interview was missed and language barriers made communication difficult", "Similar_IDs": "8"},
    {"ID": 9, "Text": "Their customer service was great They delivered to the wrong address but replaced the order without any hassle", "Similar_IDs": "9"}
]

# DataFrame erstellen
ground_truth_df = pd.DataFrame(ground_truth_data)

# Datei speichern
ground_truth_path = "/Users/john-thomas/Desktop/Seminararbeit Embedding Modelle/data/processed/ground_truth.csv"
ground_truth_df.to_csv(ground_truth_path, index=False)
print(f"Ground Truth gespeichert unter: {ground_truth_path}")
