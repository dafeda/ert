import json
from abc import ABC, abstractmethod
from typing import Any, TextIO


class Serializer(ABC):
    @abstractmethod
    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("not implemented")

    @abstractmethod
    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not implemented")

    @abstractmethod
    def encode_to_file(self, obj: Any, fp: TextIO, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("not implemented")

    @abstractmethod
    def decode_from_file(self, fp: TextIO, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not implemented")


class _json_serializer(Serializer):
    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        return json.dumps(obj, *args, **kwargs)

    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        return json.loads(series, *args, **kwargs)

    def encode_to_file(self, obj: Any, fp: TextIO, *args: Any, **kwargs: Any) -> None:
        json.dump(obj, fp)

    def decode_from_file(self, fp: TextIO, *args: Any, **kwargs: Any) -> Any:
        return json.load(fp)
