import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

# Carica e preprocessa il dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Crea il modello insegnante personalizzato
teacher_model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Compila e addestra il modello insegnante
teacher_model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
teacher_model.fit(train_images, train_labels, epochs=10)

# Crea il modello studente personalizzato
student_model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compila il modello studente
student_model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Esegue la Distillation
for epoch in range(10):
    with tf.GradientTape() as tape:
        teacher_logits = teacher_model(train_images)
        student_logits = student_model(train_images)
        loss = SparseCategoricalCrossentropy(from_logits=True)(train_labels, student_logits) + \
              SparseCategoricalCrossentropy(from_logits=True)(train_labels, teacher_logits)
    
    gradients = tape.gradient(loss, student_model.trainable_variables)
    Adam().apply_gradients(zip(gradients, student_model.trainable_variables))

# Valuta il modello studente
student_model.evaluate(test_images, test_labels)