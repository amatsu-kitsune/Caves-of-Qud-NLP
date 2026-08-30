# Caves of Qud — Ontology Schema Card

Namespace: `http://www.semanticweb.org/chris/ontologies/caverne-di-qud#`  ·  16 classes · 28 object properties · 7 datatype properties · 63 individuals

## Class hierarchy (subClassOf)
- Character
  - NPC
  - Organic
    - Mutant
    - Truekin
  - PlayerCharacter
  - Robot
- Faction
- Item
- Place
- Quest
- Race
- Role
  - Fighter
  - Trader
- Settlement

## Complete (equivalentClass) — for a reasoner, not SHACL
- Character ≡ NPC ⊔ PlayerCharacter
- Character ≡ Organic ⊔ Robot
- Organic ≡ Mutant ⊔ Truekin
- Role ≡ Fighter ⊔ Trader

## Disjoint classes
- Fighter ⊥ Trader
- Mutant ⊥ Truekin
- NPC ⊥ PlayerCharacter
- Organic ⊥ Robot

## Object properties  (domain → range)
- **containsSettlement**: Place → Settlement  (inverse: locatedInPlace)
- **dislikedBy**: Character → Faction  (inverse: dislikes)
- **dislikes**: Faction → Character
- **foughtIn**: Faction → Quest  (inverse: involvesFightWith)
- **giveQuest**: NPC → Quest  (inverse: isGivenBy)
- **hasIndividual**: Race → Mutant  (inverse: hasRace)
- **hasInhabitant**: Place ⊔ Settlement → NPC  (inverse: isHabitantOf)
- **hasMember**: Faction → NPC  (inverse: memberOf)
- **hasRace**: Mutant → Race
- **hasRole**: NPC → Role  (inverse: isRoleOf)
- **hates**: Faction → Faction
- **hostsQuest**: Place ⊔ Settlement → Quest  (inverse: takesPlaceIn)
- **involvesFightWith**: Quest → Faction
- **involvesFinding**: Quest → Item ⊔ NPC  (inverse: isSearchTargetIn)
- **isGivenBy**: Quest → NPC
- **isHabitantOf**: NPC → Place ⊔ Settlement
- **isRewardFor**: Item → Quest  (inverse: rewardsWith)
- **isRoleOf**: Role → NPC
- **isSearchTargetIn**: Item ⊔ NPC → Quest
- **likedBy**: Character → Faction  (inverse: likes)
- **likes**: Faction → Character
- **locatedInPlace**: Settlement → Place
- **memberOf**: NPC → Faction
- **requiresCompleting**: Quest → Quest
- **rewardsWith**: Quest → Item
- **sellsItem**: Trader → Item
- **spawnsIn**: PlayerCharacter → Place
- **takesPlaceIn**: Quest → Place ⊔ Settlement

## Datatype properties  (domain → type)
- **Boss**: Fighter → boolean
- **HP**: ? → literal
- **Tier**: Trader → {high, low, mid}
- **Value**: ? → literal
- **Weight**: ? → literal
- **level**: Character → integer
- **strata**: Place ⊔ Settlement → integer

## Individuals by class
- **Faction** (6): Barathrumites, Girsh, Mechanimists, Putus Templar, Seraphic Covenant, Villagers of Joppa
- **Fighter** (2): Fighter Boss, Fighter NonBoss
- **Item** (7): Copper Wire, Joppa Recoiler, Scrapped Waydroid, Sparafucile s Carbine, metamorphic polygel, neutron flux, quantum mote
- **Mutant** (11): Argyve, Asphodel Earl of Omonporch, Barathrum the Old, Golem, Otho, Pax Qlanq, Phinae Hoshaiah High Priest of the Rock, Slog of the Cloaca, Sparafucile, Starformed Ehalcodon, alchemist
- **NPC** (15): Argyve, Asphodel Earl of Omonporch, Baetyl, Barathrum the Old, Golem, Herodododicus, Otho, Pax Qlanq, Phinae Hoshaiah High Priest of the Rock, Reseph, Saad Amus, Slog of the Cloaca, Sparafucile, Starformed Ehalcodon, alchemist
- **Place** (7): Bethesda Susa, Eaters Tomb, Golgotha, Omonporch, Rainbow Wood, Rust Wells, Starfarers Quay
- **PlayerCharacter** (1): Kun
- **Quest** (11): A Call to Arms, A Canticle for Barathrum, Decoding the Signal, More Than a Willing Spirit, Pax Qlanq I Presume?, Reclamation, The Earl of Omonporch, The Golem, Tomb of the Eaters, We Are Starfreight, Weirdwire Conduit Eureka!
- **Race** (8): Fungi, Girsh Nephilim, Mollusk, Mutated Human, Robot, Sentient Plant, Truekin, Urshiib
- **Robot** (3): Baetyl, Herodododicus, Reseph
- **Settlement** (3): Grit Gate, Joppa, Temple of the Rock
- **Trader** (3): Trader high, Trader low, Trader mid
- **Truekin** (2): Kun, Saad Amus

## Constraints for generation
- Every triple MUST satisfy the property's domain and range above.
- Respect disjoint classes (an individual cannot be both, e.g. Fighter ⊥ Trader).
- Use ONLY the individuals listed; relations only between type-compatible individuals.
- Inverse pairs are equivalent facts (state either direction).