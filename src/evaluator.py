def prep_for_eval(input):
    return [[tokens_byte_pair.get(str(item), item) for item in sentence if item not in {0, 1, 2}] for sentence in input]

bleu_metric = BLEUScore(n_gram=4, smooth=True).to(device)
bleu_metric.reset()

model = Transformer(6, 512, 8, 1277, 300)
checkpoint = torch.load("./model/1.ckpt")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

pbar = tqdm(enumerate(test_loader), total = len(test_loader), desc = "Testing")
with torch.no_grad():
    for batch_idx, (src, decoder_input, targets, src_key_padding_mask, decoder_input_key_padding_mask) in pbar:
        src, decoder_input = src.to(device), decoder_input.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)
        decoder_input_key_padding_mask = decoder_input_key_padding_mask.to(device)
        outputs = model(src, decoder_input, src_key_padding_mask, decoder_input_key_padding_mask)
        predicted = outputs.argmax(dim=-1).cpu().tolist()
        predicted = prep_for_eval(predicted)
        targets = prep_for_eval(targets.tolist())
        predicted = [' '.join(tokens) for tokens in predicted]
        targets = [[' '.join(tokens)] for tokens in targets]
        bleu_metric.update(predicted, targets)

final_bleu = bleu_metric.compute()
print(f"Final BLEU: {final_bleu.item():.4f}")