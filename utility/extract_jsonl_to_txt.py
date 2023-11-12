import json

def convert_jsonl_to_text(jsonl_file_path, text_file_path):
    with open(jsonl_file_path, 'r') as jsonl_file, open(text_file_path, 'w') as text_file:
        for line in jsonl_file:
            # Analizza ogni riga JSON
            data = json.loads(line)

            # Inizializza una variabile per tenere traccia del contenuto corrente
            current_content = ""

            # Estrai i messaggi
            for message in data['messages']:
                role = message['role']
                content = message['content']

                # Aggiungi il contenuto di 'system' e 'user' a current_content
                if role == 'system':
                    current_content += " \n System Context: " +  content + "\n "
                elif role == 'user':
                    current_content += content + " "

                # Quando si arriva a 'assistant', scrivi tutto e azzera current_content
                elif role == 'assistant':
                    text_file.write(current_content.strip() + "\nRisposta: " + content + "\n\n")
                    current_content = ""

# Percorso del file JSONL di input e del file di testo di output
input_jsonl = 'data/Gata_output_231111_0959_FT3_RC_StoryKubeV1.2.jsonl'
output_text = 'output2.txt'

convert_jsonl_to_text(input_jsonl, output_text)
