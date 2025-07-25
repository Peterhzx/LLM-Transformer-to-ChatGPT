from models.original_transformer import Transformer


class TrainTransformer:
    def __init__(self, hyperparams):

        self.model = Transformer(num_layers, embed_dim, num_heads, feedforward_dim, num_tokens, words_dic["<PAD>"])

    @staticmethod
    def _check_cuda_availability():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        return device
    @staticmethod
    def reset_from_last_checkpoint():
        ckpt_files = sorted(Path("./checkpoints/transformer").glob("*.ckpt"), key=os.path.getmtime)
        if not ckpt_files:
            print("[Error] No checkpoint found to reset.")
            return 0, 0
        last_ckpt = ckpt_files[-1]
        print(f"[Reloading checkpoint] {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
        current_step = checkpoint.get('current_step', 0)
        current_epoch = checkpoint.get('current_epoch', 0)
        # current_step = 0
        # current_epoch = 0
        return current_step, current_epoch


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def text_save(self, filename, data1):
        file = open(filename, 'a')
        for i in range(len(data1)):
            s1 = str(data1[i]).replace('[', '').replace(']', '')
            s1 = s1.replace("'", '').replace(',', '') + '\n'
            file.write(s1)
        file.close()


        model.apply(self.init_weights)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-2)
        lambdalr = lambda step: embed_dim ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** (-1.5)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdalr)
        criterion = nn.CrossEntropyLoss(ignore_index=words_dic["<PAD>"],
                                        label_smoothing=0.1)  # you want to skip those 0s in loss and grad.
        current_step, current_epoch = reset_from_last_checkpoint()

        train_acc = []
        valid_acc = []
        train_loss_array = []
        valid_loss_array = []

        def train(epoch, current_step):
            print('\nEpoch: %d' % epoch)
            # torch.autograd.set_detect_anomaly(True)
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            train_iter = iter(train_loader)
            train_iter = islice(train_iter, current_step, None)
            pbar = tqdm(train_iter, total=len(train_loader), desc="Training", initial=current_step)
            for batch_idx, (src, decoder_input, targets) in enumerate(pbar,
                                                                      start=current_step):  # , src_key_padding_mask, decoder_input_key_padding_mask
                # for batch_idx, (src, decoder_input, targets, src_key_padding_mask, decoder_input_key_padding_mask) in tqdm(enumerate(train_loader)):
                # pbar.set_description(f"batch:{batch_idx}/{len(train_loader)}")
                src, decoder_input, targets = src.to(device), decoder_input.to(device), targets.to(device)
                # targets, src_key_padding_mask = targets.to(device), src_key_padding_mask.to(device)
                # decoder_input_key_padding_mask = decoder_input_key_padding_mask.to(device)
                optimizer.zero_grad()
                try:
                    outputs = model(src, decoder_input)  # , src_key_padding_mask, decoder_input_key_padding_mask)
                    if torch.isnan(outputs).any():
                        print(outputs)
                        print(loss)
                        raise ValueError("NaN in model output")
                    outputs = outputs.reshape(-1, outputs.size(-1))  # (batch*tgt_len, vocab_size)
                    targets = targets.reshape(-1)  # (batch*tgt_len,)
                    loss = criterion(outputs, targets)
                    if torch.isnan(loss):
                        print(outputs)
                        print(loss)
                        raise ValueError("NaN in loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=5.0)  # It prevents exploding gradients in deep Transformer models.
                    optimizer.step()
                    scheduler.step()
                except ValueError as e:
                    print(f"[Warning] {e} at step {batch_idx}. Reloading last checkpoint...")
                    # current_step, _ = reset_from_last_checkpoint()
                    continue  # Skip this batch
                train_loss += loss.item()
                predicted = outputs.argmax(dim=-1)
                mask = (targets != words_dic["<PAD>"])
                correct += ((predicted == targets) & mask).sum().float()
                total += mask.sum().float()
                # correct += (predicted == targets).sum().item()
                # total += targets.numel()
                if (batch_idx + 1) % 5000 == 0:
                    # text_save("./model/train_acc.txt", train_acc)
                    # text_save("./model/valid_acc.txt", valid_acc)
                    torch.save({
                        'current_epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'current_step': batch_idx + 1,
                    }, f"./model/epoch{epoch}_step{batch_idx + 1}.ckpt")

                pbar.set_postfix(loss=f"{train_loss / (batch_idx + 1):.4f}", acc=f"{correct / total:.2%}")
                # print(batch_idx, '|', len(train_loader), '|', train_loss/(batch_idx+1), '|', 100.*correct/total, '|', f"当前显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

                train_acc.append(correct / total)
                train_loss_array.append(train_loss / len(train_loader))
            # text_save("./model/train_acc.txt", train_acc)
            # text_save("./model/valid_acc.txt", valid_acc)
            """
            model_state_path = "./model/" + str(epoch) + ".ckpt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
              }, model_state_path)
              """

        def test(epoch):
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
                for batch_idx, (
                src, decoder_input, targets) in pbar:  # , src_key_padding_mask, decoder_input_key_padding_mask
                    src, decoder_input, targets = src.to(device), decoder_input.to(device), targets.to(device)
                    # targets, src_key_padding_mask = targets.to(device), src_key_padding_mask.to(device)
                    # decoder_input_key_padding_mask = decoder_input_key_padding_mask.to(device)
                    outputs = model(src, decoder_input)
                    outputs = outputs.view(-1, outputs.size(-1))  # (batch*tgt_len, vocab_size)
                    targets = targets.view(-1)  # (batch*tgt_len,)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    predicted = outputs.argmax(dim=-1)
                    correct += (predicted == targets).sum().item()
                    total += targets.numel()

                    # print(batch_idx, '|', len(val_loader), '|', test_loss/(batch_idx+1), '|', 100.*correct/total)
                    pbar.set_postfix(loss=f"{test_loss / (batch_idx + 1):.4f}", acc=f"{correct / total:.2%}")
            valid_acc.append(correct / total)
            valid_loss_array.append(test_loss / len(val_loader))

            torch.save({
                'current_epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'current_step': batch_idx+1,
            }, f"./model/epoch{epoch}.ckpt")

        for epoch in range(current_epoch, num_epochs):
            train(epoch, current_step)
            test(epoch)
            current_step = 0

        text_save("./model/train_acc.txt", train_acc)
        text_save("./model/valid_acc.txt", valid_acc)
        text_save("./model/trainloss.txt", train_loss_array)
        text_save("./model/validloss.txt", valid_loss_array)