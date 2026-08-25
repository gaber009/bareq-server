from plate_decoder import PlateDecoder

decoder = PlateDecoder()
test_inputs = [
    'أ ب ج 1234',
    'أ ب م 1234',
    'ا ب م 1234',
    'اب م 1234',
    'ابم 1234',
    'أبم 1234',
    'ألف باء ميم 1234',
    'الف باء ميم 1234',
    'الف به ميم 1234',
    'الف بيه ميم 1234',
    'ألف با ميم 1234',
    'ألف ب م 1234',
    'ا ب د 1234',
    'ابد 1234',
    'أ ب ص 1234',
    'أ م د 1234',
    'ط د ل 4782',
    'طدل 4782',
    'ر ك ع 7511',
    'ركع 7511',
    'س ش ص 4567',
    'سشص 4567',
]

for inp in test_inputs:
    dec = decoder.decode_final(inp)
    print(f"Input: {inp:<22} -> Norm: {dec['normalized']:<15} -> Plate: '{dec['plate']}' (Valid: {dec['valid']})")
