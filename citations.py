from serpapi import GoogleSearch
import requests

def main():
    params = {
        "engine": "google_scholar_cite",
        "q": "Deep Learning for AI",
        "api_key": "ac38f105d3bb09735306a159a78fab2a225b75cd43fb955733ffaa6245fa8883"
    }
    paper_names = [ "Deep learning for AI",
 "Events as changes in the layout of affordances",
 "Infogan: Interpretable representation learning by information maximizing generative adversarial nets",
 "Hierarchical reinforcement learning as creative problem solving",
 "A Unified Model of Ad Hoc Concepts in Conceptual Spaces",
 "Amalgamating evidence of dynamics",
 "Analyzing machine‐learned representations: A natural language case study",
 "A theory of the discovery and predication of relational concepts",
 "A theory of relation learning and cross-domain generalisation",
 "Mediated Generalization and Stimulus Equivalence",
 "Conceptual integration networks",
 "Why general artificial intelligence will not be realised",
 "Metacognition and reasoning",
 "GPT-3: Its nature, scope, limits, and consequences",
 "Intelligence and creativity share a common cognitive and neural basis",
 "Conceptual spaces: The geometry of thought",
 "Shortcut learning in deep neural networks",
 "How to think about perceptual learning: Twenty-five years later",
 "The theory of affordances",
 "From Computational Creativity to Creative Problem Solving Agents",
 "Nips 2016 tutorial: Generative adversarial networks",
 "Neural turing machines",
 "Transfer of learning: Cognition and instruction (p",
 "Visual affordance and function understanding: A survey",
 "Temporal updating, behavioral learning, and the phenomenology of time-consciousness",
 "Generalisation to New Actions in Reinforcement Learning",
 "Auto-encoding variational bayes",
 "Online determination of value-function structure and action-value estimates for reinforcement learning in a cognitive architecture",
 "Object-centric learning with slot attention",
 "A unified approach to interpreting model predictions",
 "Stacked convolutional auto-encoders for hierarchical feature extraction",
 "The associative basis of the creative process",
 "Memory-augmented reinforcement learning for image-goal navigation",
 "Abstraction and analogy‐making in artificial intelligence",
 "Perceptual learning in appetitive conditioning: Analysis of the Effectiveness of the Common Element",
 "Associative learning should go deep",
 "Clustergan: Latent space clustering in generative adversarial networks",
 "Rule learning by rats",
 "Why teach thinking?",
 "Cognition and the Creative Machine: Cognitive AI for Creative Problem Solving (p",
 "GOT: an optimal transport framework for graph comparison",
 "The computational origin of representation",
 "Human representation learning",
 "Automatic Data Augmentation for Generalization in Reinforcement Learning",
 "Zero-shot text-to-image generation",
 "A Generalist Agent",
 "Artificial intelligence: a modern approach (Chapter",
 "To afford or not to afford: A new formalization of affordances toward affordance-based robot control",
 "Artificial Intelligence and the Common Sense of Animals",
 "Abstraction for Deep Reinforcement Learning",
 "Apply rich psychological terms in AI with care",
 "From perceptual categories to concepts: What develops?",
 "Reinforcement learning: An introduction",
 "Grandmaster level in StarCraft II using multi-agent reinforcement learning",
 "The rhetoric and reality of anthropomorphism in artificial intelligence",
 "A preliminary framework for description, analysis and comparison of creative systems",
 "Generalization guides human exploration in vast decision spaces",
 "Scalable Gromov-Wasserstein learning for graph partitioning and matching",
 "Table to text generation with accurate content copying",
 "Reinforcement learning with prototypical representations",
 "Understanding deep learning (still) requires rethinking generalisation"]

    for paper in paper_names:

        params = {
            "engine": "google_scholar_cite",
            "q": paper,
            "api_key": "ac38f105d3bb09735306a159a78fab2a225b75cd43fb955733ffaa6245fa8883"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        if "links" in results:
            citations = results["links"]

            link = citations[0]['link']
            f = requests.get(link)
            print(f.text)


if __name__ == '__main__':
    main()
