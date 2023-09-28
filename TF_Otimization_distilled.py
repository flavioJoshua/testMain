import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

# Carica e preprocessa il dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Crea il modello insegnante personalizzato
teacher_model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu'),
    Dense(10)  # No activation here, since we're using from_logits=True
])

# Compila e addestra il modello insegnante
teacher_model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
teacher_model.fit(train_images, train_labels, epochs=10)

# Crea il modello studente personalizzato
student_model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(128, activation='relu'),
    Dense(10)  # No activation here, since we're using from_logits=True
])

# Compila il modello studente
student_model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Definizione delle funzioni di perdita e dell'ottimizzatore
scce_loss = SparseCategoricalCrossentropy(from_logits=True)
kl_loss = KLDivergence()
optimizer = Adam()

# Esegue la Distillation
for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    
    # Utilizzo di GradientTape per registrare le operazioni da cui calcolare i gradienti
    with tf.GradientTape() as tape:
        # Ottiene i logit dal modello insegnante e dal modello studente
        teacher_logits = teacher_model(train_images)
        student_logits = student_model(train_images)
        
        # Calcola la perdita di classificazione
        classification_loss = scce_loss(train_labels, student_logits)
        
        # Calcola la perdita di distillazione
        distillation_loss = kl_loss(tf.nn.softmax(teacher_logits / 2), tf.nn.softmax(student_logits / 2))
        
        # Combina le perdite
        loss = classification_loss + distillation_loss
    
    # Calcola i gradienti
    gradients = tape.gradient(loss, student_model.trainable_variables)
    
    # Applica i gradienti per aggiornare i pesi del modello studente
    optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

# Valuta il modello studente
test_loss, test_accuracy = student_model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy * 100}%")
