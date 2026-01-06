from threading import Lock

MAX_IMGS_BYTES = 128 * (1024) * (1024)

class ImgCache:
    def __init__(self, max_size: int = MAX_IMGS_BYTES):
        self._container = {}
        self._max_size = max_size
        self._actual_size = 0
        self._lock = Lock()

    def __getitem__(self, uuid: str):
        return self._container.get(uuid, None)
    
    def add_img(self, uuid: str, img: bytes, content_type: str, date: float):
        with self._lock:
            incr_size = len(img)

            stackedkeys = sorted(
                [(key, value["date"]) for key, value in self._container.items()],
                key=lambda x: x[1]
            )
            
            while self._actual_size + incr_size > self._max_size and stackedkeys:
                self.remove_img(stackedkeys.pop(0)[0])

            self._container[uuid] = {
                "content": img,
                "content-type": content_type,
                "date": date, 
            }
            self._actual_size += incr_size

    def remove_img(self, uuid: str):
        with self._lock:
            size = len(self._container[uuid]["content"])
            del self._container[uuid]
            self._actual_size -= size