# SampleModel class is the corresponding class of Sample Table in the SQL database
class SampleModel:

    # academic_id references AcademicModel class, int type
    # sample_photo is PIL Image
    # date is datetime.datetime
    # location object type will be compatible with GoogleMaps API
    # publication_status and anonymous_status are booleans
    # pollens is a dictionary {pollen_type: number}
    def __init__(self, sample_id, academic_id, date, location_latitude, location_longitude, analysis_text, publication_status, anonymous_status, pollens):
        self.sample_id = sample_id
        self.academic_id = academic_id
        self.date = date
        self.location_latitude = location_latitude
        self.location_longitude = location_longitude
        self.analysis_text = analysis_text
        self.publication_status = publication_status
        self.anonymous_status = anonymous_status
        self.pollens = pollens
