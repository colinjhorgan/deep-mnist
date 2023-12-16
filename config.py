import os 

from torchvision import datasets


def get_mnist():
    """Gets the MNIST dataset from torchvision.datasets
    and saves it to a newly created directory 'data'.
    Deletes all compressed artifacts after successful download.
    """
    datasets.MNIST(root='data', download=True)
    path = "./data/MNIST/raw/"
    contents = os.listdir(path)
    for item in contents:
        if item.endswith(".gz"):
            os.remove(os.path.join(path, item)) 
            print(f"Deleted {item}")


if __name__ == "__main__":
    get_mnist()
