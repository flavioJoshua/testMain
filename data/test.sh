curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{
            "role": "system",
            "content": "Translate the following English text to Italian:"
        },
        {
            "role": "user",
            "content": "The cat is on the table."
        }],
     "temperature": 0.7
   }'