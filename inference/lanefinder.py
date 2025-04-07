import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters.classify import set_input, get_classes
from image.processing import preprocessing, postprocessing
from PIL import Image

class Lanefinder:

    def __init__(self, model, input_shape, output_shape, quant, dequant):
        self._window = None
        self._interpreter = self._get_tpu_interpreter(model)
        self._cap = cv2.VideoCapture(0)
        self._size = input_shape
        self._output_shape = output_shape
        self._quant = quant
        self._dequant = dequant

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, name):
        self._window = name

    @staticmethod
    def _get_tpu_interpreter(model):
        try:
            interpreter = make_interpreter(model)
            interpreter.allocate_tensors()
        except RuntimeError:
            interpreter = None
        return interpreter

    def _preprocess(self, frame):
        return preprocessing(frame, self._quant['mean'], self._quant['std'])

    def _postprocess(self, pred_obj, frame):
        return postprocessing(
            pred_obj=pred_obj,
            frame=frame,
            mean=self._quant['mean'],
            std=self._quant['std'],
            in_shape=self._size,
            out_shape=self._output_shape
        )

    def stream(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            frame = np.array(frame)
            frmcpy = frame.copy()

            frame_resized = cv2.resize(frame, tuple(self._size))
            frame_resized = frame_resized.astype(np.float32)

            if self._interpreter is not None:
                frame_preprocessed = self._preprocess(frame_resized)
                input_tensor_index = self._interpreter.get_input_details()[0]['index']
                self._interpreter.set_tensor(input_tensor_index, frame_preprocessed)
                self._interpreter.invoke()
                output_tensor_index = self._interpreter.get_output_details()[0]['index']
                pred_obj = self._interpreter.get_tensor(output_tensor_index)
                pred = self._postprocess(pred_obj, frmcpy)
            else:
                frmcpy = cv2.resize(frmcpy, self._output_shape)
                pred = cv2.putText(
                    frmcpy,
                    'TPU has not been detected!',
                    org=(self._output_shape[0] // 16, self._output_shape[1] // 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=1
                )

            window_name = self._window if self._window else 'default'
            cv2.imshow(window_name, pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def destroy(self):
        cv2.destroyAllWindows()
        self._cap.release()

class LanefinderFromVideo(Lanefinder):

    def __init__(self, src, model, input_shape, output_shape, quant, dequant):
        super().__init__(model, input_shape, output_shape, quant, dequant)
        self._cap = cv2.VideoCapture(src)
