import matplotlib.pyplot as plt
from baseline_models.animalai_loader import AnimalAIEnvironmentLoader


def load_env():
    animal_ai_loader = AnimalAIEnvironmentLoader()
    env = animal_ai_loader.get_animalai_env()
    obs = env.reset()
    return obs, env


def only_fwd():
    obs, env = load_env()
    plt.imshow(obs)
    plt.show()
    for i in range(0, 400):
        new_img, rwd, _, _ = env.step(3)
        plt.imshow(new_img)
        plt.show()
        print(rwd)


def only_bwd():
    obs, env = load_env()
    plt.imshow(obs)
    plt.show()

    for i in range(0, 9):
        new_img, rwd, _, _ = env.step(8)
        plt.imshow(new_img)
        plt.show()


def all_action():
    obs, env = load_env()
    plt.imshow(obs)
    plt.show()
    for i in range(0, 9):
        new_img, _, _, _ = env.step(i)
        plt.imshow(new_img)
        plt.show()


if __name__ == '__main__':
    only_fwd()

