import json

def cleanup(row):
    if row['entity_sentiments'] is None or row['entity_sentiments'] == '':
        return {'entity_sentiments': None}
    entity_sentiments = json.loads(row['entity_sentiments'])
    if isinstance(entity_sentiments, list):
        entity_sentiments = entity_sentiments[0]
    try:
        if 'entities' in entity_sentiments.keys() and isinstance(entity_sentiments['entities'], list):
            dct = {}
            for entity in entity_sentiments['entities']:
                if 'entity' in entity.keys() and 'sentiment' in entity.keys():
                    dct[entity['entity']] = entity['sentiment']
                if 'name' in entity.keys() and 'sentiment' in entity.keys():
                    dct[entity['name']] = entity['sentiment']
                if len(entity) == 1 and isinstance(list(entity.values())[0], int) and isinstance(list(entity.keys())[0], str):
                    dct[list(entity.keys())[0]] = list(entity.values())[0]
            return {'entity_sentiments': json.dumps(dct)}
        else:
            return {'entity_sentiments' :json.dumps(entity_sentiments)}
    except Exception as e:
        print(f"Error processing row: {entity_sentiments}")
        print(f"Exception: {e}")
        return {'entity_sentiments': json.dumps(entity_sentiments)}