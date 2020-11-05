import collections
import random
import code

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        """
        Initialize the card.

        Args:
            self: (todo): write your description
        """
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        """
        Returns the number of bytes.

        Args:
            self: (todo): write your description
        """
        return len(self._cards)

    def __getitem__(self, position):
        """
        Return a single item from the given position.

        Args:
            self: (todo): write your description
            position: (todo): write your description
        """
        return self._cards[position]

if __name__ == '__main__':
    deck = FrenchDeck()
    card = random.choice(deck)
    code.interact(local=locals())
