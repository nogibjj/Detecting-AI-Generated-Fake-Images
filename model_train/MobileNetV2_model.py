import boto3
import tensorflow.compat.v2 as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from umap import UMAP
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

tf.keras.backend.set_image_data_format('channels_first')

s3 = boto3.resource('s3')
bucket_name = 'dalle2images'
real_image_paths = []
fake_image_paths = []
real_labels = []
fake_labels = []

# loop through the images in your S3 bucket and append them to their respective lists:
for obj in s3.Bucket(bucket_name).objects.all():
    if obj.key.endswith('.png') and obj.key.startswith('real'):
        real_image_paths.append(obj.key)
        real_labels.append(0)
    elif obj.key.endswith('.png') and obj.key.startswith('fake'):
        fake_image_paths.append(obj.key)
        fake_labels.append(1)
        
# randomly select 100 images (50 real, 50 fake) from your bucket
n_images = 100
n_real = n_fake = n_images // 2
real_samples = np.random.choice(real_image_paths, n_real, replace=False)
fake_samples = np.random.choice(fake_image_paths, n_fake, replace=False)
sample_paths = np.concatenate([real_samples, fake_samples])

images = []
orig_images = []  # Added: Store original images before preprocessing
labels = []
image_size = (224, 224)

for i, path in enumerate(sample_paths):
    img = s3.Object(bucket_name, path)
    img = Image.open(img.get()['Body'])
    img = img.resize(image_size, Image.ANTIALIAS)

    orig_img = np.array(img)  # Added: Store original image before preprocessing
    orig_images.append(orig_img)  # Added: Append original image to the list

    img = np.array(img).transpose((2, 0, 1))  # Change image format to channels_first
    img = img[..., ::-1]  # Convert from RGB to BGR format
    img = preprocess_input(img)

    images.append(img)
    labels.append(0 if path.startswith('real') else 1)

images = np.stack(images)
orig_images = np.stack(orig_images)  # Added: Stack original images
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_orig_images, test_orig_images, _, _ = train_test_split(orig_images, labels, test_size=0.2, random_state=42)  # Added: Split original images

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_dataset = train_dataset.shuffle(buffer_size=len(train_images), reshuffle_each_iteration=True)

batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Load the pre-trained MobileNetV2 model without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(3, 224, 224))



# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Add the data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

# Define the regularization strength
reg_strength = 0.001

# Create a new model using the base model and adding custom top layers
model = tf.keras.models.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(reg_strength)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(reg_strength)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(reg_strength)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print("Test accuracy: {:.2f}%".format(accuracy * 100))

# # Extract the features from the trained model for t-SNE visualization
# features = model.predict(test_images)

# umap = UMAP(n_components=2, random_state=42)
# umap_features = umap.fit_transform(features)

# import matplotlib.pyplot as plt
# import matplotlib.offsetbox as offsetbox


# fig, ax = plt.subplots(figsize=(10, 10))

# # Scaling factor to control the size of the displayed images
# scale_factor = 0.1
# border_thickness = 1

# # Modify the visualization loop to use test_orig_images instead of test_images
# for i, (x, y) in enumerate(umap_features):
#     img = test_orig_images[i]  # Load the original image
    
#     # Resize the image and add a colored border
#     img_resized = Image.fromarray(img).resize((int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor)), Image.ANTIALIAS)
#     border_color = 'red' if test_labels[i] == 1 else 'green'
#     bordered_image = Image.new('RGB', (img_resized.width + 2 * border_thickness, img_resized.height + 2 * border_thickness), border_color)
#     bordered_image.paste(img_resized, (border_thickness, border_thickness))
    
#     # Add the image to the plot
#     imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(bordered_image), (x, y), frameon=False)
#     ax.add_artist(imagebox)

# # Set the plot limits and display the plot
# ax.set_xlim(umap_features[:, 0].min() - 5, umap_features[:, 0].max() + 5)
# ax.set_ylim(umap_features[:, 1].min() - 5, umap_features[:, 1].max() + 5)
# plt.show()
