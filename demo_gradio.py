from tensorflow import keras
import numpy as np
import gradio as gr
import tensorflow.compat.v2 as tf

def main():

    model = keras.models.load_model(args.checkpoint, compile=False)

    def segment_image(inp):
        inp = tf.expand_dims(inp, axis=0)
        output = model.predict(inp / 255)
        output = np.where(output > args.threshold, output, 0)
        return output[0, :, :, 0]

    demo = gr.Interface(fn=segment_image,
                        inputs=[gr.Image(shape=(args.image_size, args.image_size))],
                        outputs="image")
    demo.launch()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, help='A path to checkpoint', required=True)
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training and dataloader')
    parser.add_argument('--threshold', type=float, default=0.5, help='A threshold for the predicted mask')
    args = parser.parse_args()

    main()

    #python demo_gradio.py --checkpoint checkpoints/inception_resnetv2_unet.h5