from enum import Enum

# FeedbackModel class is the corresponding class of Feedback Table in the SQL database
class FeedbackModel:

    # ids are int
    # date is datetime.datetime
    # status has its own Enum
    def __init__(self, feedback_id, academic_id, name, email, text, date, status):
        self.feedback_id = feedback_id
        self.academic_id = academic_id
        self.name = name
        self.email = email
        self.text = text
        self.date = date
        self.status = status


# Enum class for FeedbackModel.status
class FeedbackModelStatus(Enum):
    pending = 1
    answered = 2
