from typing import List, Dict, Union
import torch
from art.estimators.object_detection import PyTorchObjectDetector
import numpy as np


class ModifyPyTorchObjectDetector(PyTorchObjectDetector):
    def loss_gradient(  # pylint: disable=W0613
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:
                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        grad_list = []

        # Adding this loop because torch==[1.7, 1.8] and related versions of torchvision do not allow loss gradients at
        #  the input for batches larger than 1 anymore for PyTorch FasterRCNN because of a view created by torch or
        #  torchvision. This loop should be revisited with later releases of torch and removed once it becomes
        #  unnecessary.
        for i in range(x.shape[0]):

            x_i = x[[i]]
            y_i = [y[i]]

            output, inputs_t, image_tensor_list_grad = self._get_losses(
                x=x_i, y=y_i)

            # Compute the gradient and return
            loss = None
            for loss_name in output.keys():
                if loss is None:
                    loss = output[loss_name]
                else:
                    loss = loss + output[loss_name]

            # Clean gradients
            self._model.zero_grad()

            # Compute gradients
            loss.backward(retain_graph=True)  # type: ignore

            if isinstance(x, np.ndarray):
                for img in image_tensor_list_grad:
                    gradients = img.grad.cpu().numpy().copy()
                    grad_list.append(gradients)
            else:
                for img in inputs_t:
                    gradients = img.grad.copy()
                    grad_list.append(gradients)

        if isinstance(x, np.ndarray):
            grads = np.stack(grad_list, axis=0)
            grads = np.transpose(grads, (0, 2, 3, 1))
        else:
            grads = torch.stack(grad_list, dim=0)
            grads = grads.premute(0, 2, 3, 1)

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads
