"""
ResidualBlocks.py
=================

TensorFlow version that implements a pre-activation residual block,
similar to the PyTorch version in QCN paper.

This block can be used as a building block in deeper networks. In the original
PyTorch code, pre-activation is utilized, meaning each convolution is preceded
by batch normalization and ReLU.

Ex Usage:
--------------
    from ResidualBlocks import BasicPreActResBlock
    import tensorflow as tf

    # For input of shape (batch, height, width, channels)
    x = tf.random.normal((8, 128, 128, 64))

    # Create one residual block that outputs 128 channels while performing
    # a stride of 2 on the first convolution, downsampling.
    block = BasicPreActResBlock(in_channels=64, out_channels=128, stride=2)

    # Forward pass (training=True if in training mode)
    y = block(x, training=True)
    print(y.shape)  # -> (8, 64, 64, 128)
"""

import tensorflow as tf

class BasicPreActResBlock(tf.keras.layers.Layer):
    """
    A Pre-Activation Residual Block as in the paper by He et al.
    This block uses:
        BN -> ReLU -> Conv2D -> BN -> ReLU -> Conv2D
    and then adds a skip connection from the input. If `stride != 1` or
    if the number of channels changes, a 1x1 convolution is applied to
    the skip connection for dimension matching.

    Attributes:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels produced by this block.
    stride : int
        Stride to use in the first convolution of the block. The second
        convolution always has stride=1.

    Methods:
    --------
    call(x, training=False):
        Defines the forward pass. Applies batch norm -> relu -> conv,
        batch norm -> relu -> conv, adds skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initializes the BasicPreActResBlock.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Desired number of channels in the output tensor.
        stride : int, optional
            Convolution stride for the first 3x3 conv, by default 1.
        """
        super(BasicPreActResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Batch normalization layers
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            strides=stride,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )

        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=1,
                strides=stride,
                padding='valid',
                use_bias=False,
                kernel_initializer='he_normal'
            )
        else:
            self.shortcut = None

    def call(self, x, training=False):
        """
        Forward pass of the Pre-Activation Residual Block.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, height, width, in_channels).
        training : bool, optional
            Whether the layer should behave in training mode or inference
            mode. This is passed to the batch normalization layers.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch, new_height, new_width, out_channels).
        """

        # First conv
        out = self.bn1(x, training=training)
        out = tf.nn.relu(out)
        out = self.conv1(out)

        # Second conv
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)

        # Skip-connection
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        # Add skip connection
        out += shortcut
        return out


