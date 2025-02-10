Lemma plus_comm : forall a b : nat, a + b = b + a.
Proof.
  intros a b.
  induction a.
  - simpl. rewrite Nat.add_0_r. reflexivity.
  - simpl. rewrite IHa. reflexivity.
Qed.
