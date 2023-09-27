import pickle
from aicrowd_gym.serializers.base import BaseSerializer


class PickleSerializer(BaseSerializer):

    def __init__(self):
        self.content_type = "application/octet-stream"

    def raw_encode(self, data):
        return pickle.dumps(data)

    def raw_decode(self, data):
        return pickle.loads(data)
