import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:
    _ORT_AVAILABLE = False


class YOLOv8Detector:
    def __init__(self, model_path, execution_provider="CPUExecutionProvider", confidence_threshold=0.25, iou_threshold=0.45, img_size=1280):
        if not _ORT_AVAILABLE:
            raise ImportError("onnxruntime is required for YOLOv8Detector. Install requirements.txt first.")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [execution_provider]
        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

        input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        self.input_width = self.input_shape[2] if len(self.input_shape) >= 3 else img_size
        self.input_height = self.input_shape[3] if len(self.input_shape) >= 4 else img_size

    def detect(self, frame):
        input_tensor, ratio, pad = self._preprocess(frame)
        outputs = self.session.run([self.output_name], {self.session.get_inputs()[0].name: input_tensor})
        predictions = self._postprocess(outputs[0], ratio, pad, frame.shape)
        return predictions

    def _preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, ratio, pad = self._letterbox(img, new_shape=(self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0
        return img, ratio, pad

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def _postprocess(self, outputs, ratio, pad, original_shape):
        predictions = outputs[0]
        results = []

        if predictions is None or len(predictions) == 0:
            return results

        if predictions.ndim == 2:
            predictions = np.expand_dims(predictions, 0)

        for pred in predictions:
            if pred.ndim == 1:
                pred = np.expand_dims(pred, 0)

            if pred.shape[1] == 0:
                continue

            boxes = pred[:, :4]
            scores = pred[:, 4]
            classes = pred[:, 5].astype(int)

            keep = scores >= self.confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            if len(boxes) == 0:
                continue

            boxes = self._scale_coords(boxes, ratio, pad, original_shape)
            boxes = self._xywh2xyxy(boxes)

            for i in range(len(boxes)):
                results.append({
                    "bbox": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "class": int(classes[i])
                })

        return results

    def _scale_coords(self, boxes, ratio, pad, original_shape):
        gain = ratio[0]
        pad_x = pad[0]
        pad_y = pad[1]

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, :4] /= gain

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])
        return boxes

    def _xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
