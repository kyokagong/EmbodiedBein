from eb.adapters.base import Adapter


class UfoNativeBaselineAdapter(Adapter):
    name = "ufo_native_baseline"

    def render(self, spec, run_id):
        from eb.ufo_native_baseline_service import render

        return render(spec, run_id)
