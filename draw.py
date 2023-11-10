import re
import matplotlib.pyplot as plt


def extract_loss_values(log_text):
    loss_pattern = re.compile(r'Epoch \d+/\d+: loss->([\d.]+)')
    valid_loss_pattern = re.compile(r'Epoch \d+/\d+: valid_loss->([\d.]+)')

    loss_values = loss_pattern.findall(log_text)
    valid_loss_values = valid_loss_pattern.findall(log_text)

    return loss_values, valid_loss_values


filepath = 'journal/FunkSVD_model1.txt'
logstr = ''
with open(filepath, 'r') as f:
    logstr = f.read()

loss_values, valid_loss_values = extract_loss_values(logstr)
loss_values = [float(lossvalue) for lossvalue in loss_values]
valid_loss_values = [float(lossvalue) for lossvalue in valid_loss_values]

x = [i for i in range(len(loss_values))]

plt.plot(x, loss_values, label='Training Loss')
plt.plot(x, valid_loss_values, label='Validation Loss', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.savefig(f"journal/FunkSVD_train.png")
plt.show()

