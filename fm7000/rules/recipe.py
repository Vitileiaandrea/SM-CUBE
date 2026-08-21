"""Recipe manager - enforces the arrosticini composition sequence."""


from fm7000.config.constants import CUBE, RECIPE_SEQUENCE, MeatType


class RecipeManager:
    """
    Manages the layer-by-layer recipe for cube filling.

    HIGH_QUALITY -> FAT -> MEDIUM -> FAT -> ... -> HIGH_QUALITY

    Il numero di strati non e' fisso: viene pianificato sullo spessore reale
    misurato dal profilatore, in modo che il primo e l'ultimo strato siano
    sempre alta qualita e che l'ultimo strato entri nei 250 mm del cubo.
    """

    def __init__(
        self,
        sequence: list[MeatType] | None = None,
        cube_height_mm: float = CUBE.height_mm,
    ) -> None:
        self.sequence = list(sequence) if sequence else list(RECIPE_SEQUENCE)
        self.cube_height_mm = cube_height_mm
        self.current_layer: int = 0
        self._final_forced: bool = False

    # ------------------------------------------------------------- planning

    @staticmethod
    def build_sequence(n_layers: int) -> list[MeatType]:
        """Alta qualita in apertura e chiusura, alternanza grasso/media al centro."""
        n = max(3, int(n_layers))
        middle: list[MeatType] = []
        for i in range(n - 2):
            middle.append(MeatType.FAT if i % 2 == 0 else MeatType.MEDIUM_QUALITY)
        return [MeatType.HIGH_QUALITY] + middle + [MeatType.HIGH_QUALITY]

    def plan_for_thickness(self, avg_layer_thickness_mm: float) -> list[MeatType]:
        """Pianifica gli strati sullo spessore medio compresso realmente misurato."""
        if avg_layer_thickness_mm <= 0.0:
            return self.sequence
        n_layers = int(self.cube_height_mm // avg_layer_thickness_mm)
        self.sequence = self.build_sequence(n_layers)
        return self.sequence

    # -------------------------------------------------------------- runtime

    @property
    def current_meat_type(self) -> MeatType:
        if self._final_forced:
            return MeatType.HIGH_QUALITY
        if self.current_layer < len(self.sequence):
            return self.sequence[self.current_layer]
        return MeatType.HIGH_QUALITY

    @property
    def is_first_layer(self) -> bool:
        return self.current_layer == 0

    @property
    def remaining_layers(self) -> int:
        return max(0, len(self.sequence) - self.current_layer)

    def advance_layer(self) -> MeatType:
        self.current_layer += 1
        return self.current_meat_type

    def get_required_type_for_layer(self, layer_index: int) -> MeatType:
        if layer_index < len(self.sequence):
            return self.sequence[layer_index]
        return MeatType.HIGH_QUALITY

    def fits_in_cube(
        self,
        current_height_mm: float,
        slice_thickness_mm: float,
        cube_height_mm: float | None = None,
    ) -> bool:
        """La fetta entra senza sfondare il cubo?"""
        limit = cube_height_mm or self.cube_height_mm
        return current_height_mm + slice_thickness_mm <= limit

    def should_switch_to_final(
        self,
        current_height_mm: float,
        expected_layer_thickness_mm: float = 30.0,
        cube_height_mm: float | None = None,
    ) -> bool:
        """
        Passa all'alta qualita finale quando lo spazio residuo basta solo per
        un ultimo strato: se si aspetta troppo, l'ultimo boccone non e' pregiato.
        """
        limit = cube_height_mm or self.cube_height_mm
        remaining = limit - current_height_mm
        if remaining <= 0:
            return True
        return remaining < expected_layer_thickness_mm * 2.0

    def force_final_layer(self) -> None:
        """Blocca la ricetta sull'alta qualita fino alla chiusura del cubo."""
        self._final_forced = True

    @property
    def in_final_phase(self) -> bool:
        return self._final_forced

    def is_final_layer(
        self,
        estimated_height_mm: float,
        cube_height_mm: float | None = None,
    ) -> bool:
        limit = cube_height_mm or self.cube_height_mm
        return (limit - estimated_height_mm) < 30.0

    def get_conveyor_lane(self) -> int:
        meat_type = self.current_meat_type
        if meat_type == MeatType.HIGH_QUALITY:
            return 0
        if meat_type == MeatType.MEDIUM_QUALITY:
            return 1
        return 2

    @staticmethod
    def lane_for_type(meat_type: MeatType) -> int:
        if meat_type == MeatType.HIGH_QUALITY:
            return 0
        if meat_type == MeatType.MEDIUM_QUALITY:
            return 1
        return 2

    def reset(self) -> None:
        self.current_layer = 0
        self._final_forced = False
