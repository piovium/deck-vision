# GOAL

Implement `deck-vision` project: A program that convert deck sharing image of Genius Invokation TCG (GI-TCG), to a machine-readable format, and finally the copyable share code.

## Background knowledge

- A Deck of GI-TCG contains 33 cards (3 character cards and 30 action cards).
- Each card has an internal ID, a share ID, and a card face with ratio of 7:12.
- A Deck represented in internal ID can be used in multiple project (e.g. game simulator & information collector), and a Base64-ed string, called "Share Code", can be computed from there share ID. One deck can have multiple share code with different initial seed (see below).

## Input and Output (Phase 1)

- The input of `deck-vision` is a user-uploaded image, which *may* contains cardfaces of a GI-TCG deck.
- The output of `deck-vision` is whether: success with a JSON-format object, with following field:
  ```ts
  interface Output {
    characters: number[]; // must be 3 items, internal ID of the deck's character card
    cards: number[];      // must be 30 items, internal ID of the deck's character card
    code: string;         // the share code of this deck.
  }
  ```
  or failed with detailed error message.

## Technical information

### Internal ID, Share ID, and Card face fetching

- The environment variable `ASSETS_API_ENDPOINT`, defaults to `https://static-data.piovium.org/api/v4`, should be used by program in following ways:

- `${ASSETS_API_ENDPOINT}/data/latest/CHS/characters` and `${ASSETS_API_ENDPOINT}/data/latest/CHS/action_cards`, both returns:
  ```ts
  interface CharacterOrActionCardsData {
    success: true,
    data: Array<{
      id: number;      // internal ID
      shareId?: number // share ID, only exists on card that can be used in a deck
      // [rest fields omitted]
    }>;
    cardFace: string; // a card face filename (see below)
  }
  ```
- For each card face filename `cardFace`, access `${ASSETS_API_ENDPOINT}/image/raw/${cardFace}` to download the card face image (might be in WebP or PNG format).

- The `data` array in API response is mutable by extending items along as game version update. `id`, `shareId`, `cardFace`, and image of `cardFace` are unlikely changed.

### Share code generation

The Share ID of 33 cards formed an array of 33 integers, a bitstring can be computed with this array `arr` and a initial uint8_t `seed`:

```js
function encode(arr, seed) {
  const padded = [...arr, 0];
  const reordered = Array.from({ length: 17 }).flatMap((_, i) => [
    padded[i * 2]! >> 4,
    ((padded[i * 2]! & 0xf) << 4) + (padded[i * 2 + 1]! >> 8),
    arr[i * 2 + 1]! & 0xff,
  ]);
  const original = Array.from({ length: 25 }).flatMap((_, i) => [
    (reordered[i]! + last) & 0xff,
    (reordered[i + 25]! + last) & 0xff,
  ]);
  return new Uint8Array([...original, last]);
}
```

And a valid Share Code is a base64-ed above bitstring, that contains **none** of the block-words specified in `data/block_words.txt`. The program can try generation of Share Code by enumerating `seed` from `0` to `255`.

## Example 

I'd provided several example of expected user input and outputs under `examples` folder.

## Additional requirement

- We should not rely on remote or heavy compute resources, e.g. LLM providers.
- Use of S3, Redis, or similar serverless fundamentals are acceptable.
- You can feel free to use any programming language, algorithm, and framework.

## Further steps

- In Phase 1, we can only implement a CLI or even a programmatic API only.
- For further steps, we can:
  - provide a HTTP API server for serving this functionality, and bundled into a Docker image.
  - In a failed path, the program can point out the detail of failure by marking up the input image.

