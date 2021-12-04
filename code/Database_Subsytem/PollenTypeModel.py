# PollenTypeModel class is the corresponding class of PollenType Table in the SQL database
class PollenTypeModel:

    def __init__(self, pollen_name, explanation_text):
        self.pollen_name = pollen_name
        self.explanation_text = explanation_text
