(* Fundamental Theorem of Calculus in Coq *)
Require Import Reals.
Require Import Coquelicot.Coquelicot.

Lemma FTC : forall (f F : R -> R) (a b : R),
  (forall x, a <= x <= b -> derivable_pt F x) ->
  (forall x, a <= x <= b -> derive_pt F x = f x) ->
  integral a b f = F b - F a.
Proof.
  intros f F a b Hderivable Hderive.
  apply Riemann_integral_FTC.
  - exact Hderivable.
  - exact Hderive.
Qed.
