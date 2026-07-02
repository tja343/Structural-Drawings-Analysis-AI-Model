from app.models.ocr.service import OCRService


def test_paddle_device_falls_back_to_cpu_when_gpu_unavailable(monkeypatch):
    class FakeCuda:
        @staticmethod
        def device_count():
            return 0

    class FakeDevice:
        cuda = FakeCuda()

    class FakePaddle:
        device = FakeDevice()

        @staticmethod
        def is_compiled_with_cuda():
            return False

    service = OCRService.__new__(OCRService)
    service.use_gpu = True
    monkeypatch.setitem(__import__("sys").modules, "paddle", FakePaddle)

    assert service._resolve_paddle_device() == "cpu"
    assert service.use_gpu is False
