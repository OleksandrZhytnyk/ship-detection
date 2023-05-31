from datasets.dataloader import AirbusShipSequence, aug_with_crop
from models.vgg19_unet import build_vgg19_unet
from models.inception_resnetv2_unet import build_inception_resnetv2_unet
from losses.weighted_bce_dice_loss import weighted_bce_dice_loss, dice_coef
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import os
import tensorflow as tf
from utils.utils import plot_training_history


def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load images and masks from folders
    path_to_masks = os.listdir(os.path.join(args.mask_folder))
    full_path_to_images = [args.image_folder + i.split('_')[0] + '.jpg' for i in path_to_masks]
    full_path_to_mask = [args.mask_folder + i for i in path_to_masks]
    # Split data on train and validation dataset
    X_train, y_train = full_path_to_images[:40000], full_path_to_mask[:40000]
    X_valid, y_valid = full_path_to_images[40000:], full_path_to_mask[40000:]

    print("train len ", len(X_train))
    print("valid len ", len(X_valid))
    # Create dataloader for train and validation data
    train_dataset = AirbusShipSequence(path_to_images=X_train, path_to_masks=y_train,
                                       img_size=(args.image_size, args.image_size), augmentation=aug_with_crop,
                                       batch_size=args.batch_size, use_mask=True, test_images = False)

    valid_dataset = AirbusShipSequence(path_to_images=X_valid, path_to_masks=y_valid,
                                       img_size=(args.image_size, args.image_size),
                                       batch_size=args.batch_size, use_mask=True, test_images = False)
    # Delete unused variables
    del X_train, X_valid, y_train, y_valid, full_path_to_images, full_path_to_mask

    # Model learning rate reducer
    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                   cooldown=1,
                                   patience=2, verbose=1,
                                   min_lr=0.1e-6)
    # Model autosave callbacks
    mode_autosave = ModelCheckpoint(args.save_dir + "_" + args.save_name,
                                    monitor='val_dice_coef',
                                    mode='max', save_best_only=True, verbose=1)

    tensorboard = TensorBoard(log_dir=args.logs, histogram_freq=0,
                              write_graph=True, write_images=False)
    # Stop learning as metric on validation stop increasing
    early_stopping = EarlyStopping(patience=args.patience, verbose=1, mode='auto')

    callbacks = [mode_autosave, lr_reducer, tensorboard, early_stopping]

    # Initialize model
    input_shape = (args.image_size, args.image_size, 3)
    model = build_inception_resnetv2_unet(input_shape)

    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    model.compile(optimizer=opt,
                  loss=weighted_bce_dice_loss, metrics=[dice_coef])
    # Start training
    print("start training")
    history = model.fit(train_dataset, batch_size=1, epochs=args.epochs,
                        validation_data=valid_dataset, callbacks=callbacks)

    plot_training_history(history)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, help='A path to the folder for saving checkpoints', required=True)
    parser.add_argument('--save_name', type=str, help='A model name for saving checkpoints', required=True)
    parser.add_argument('--image_folder', type=str, help='A path to images folder', required=True)
    parser.add_argument('--mask_folder', type=str, help='A path to masks folder', required=True)
    parser.add_argument('--logs', type=str, help='The TensorBoard logs', required=True)
    parser.add_argument('--epochs', type=int, default=100, help='A maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='A learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=5, help='A number of epochs to do early stoper')
    parser.add_argument('--weight_decay', type=float, default=0.004, help='A weight decay for optimizer')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training and dataloader')

    args = parser.parse_args()

    main()
