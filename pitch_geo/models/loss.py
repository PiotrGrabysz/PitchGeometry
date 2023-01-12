import tensorflow as tf


mse = tf.keras.losses.MeanSquaredError(reduction="none")
binary_crossentropy = tf.keras.losses.BinaryCrossentropy()


def create_teacher_forced_loss(weight=0.5):
    def loss_fn(y_true, y_pred):
        visible_mask = y_true[:, :, 2] != 0

        y_true_kps = y_true[:, :, :2]
        keypoint_loss = mse(y_true[:, :, :2], y_pred[:, :, :2])

        # Teacher forcing
        keypoint_loss = tf.where(
            visible_mask, keypoint_loss, tf.zeros_like(keypoint_loss, dtype="float32")
        )
        keypoint_loss = tf.math.reduce_mean(keypoint_loss)

        visibility_loss = binary_crossentropy(y_true[:, :, 2], y_pred[:, :, 2])

        total_loss = weight * keypoint_loss + (1 - weight) * visibility_loss
        return total_loss

    return loss_fn
