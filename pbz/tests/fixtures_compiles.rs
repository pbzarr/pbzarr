mod fixtures;

#[test]
fn fixtures_empty_compiles_and_creates_store() {
    let f = fixtures::StoreFixture::empty();
    assert!(f.path.exists());
}

#[test]
fn fixtures_uint32_track_compiles() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    assert!(f.path.exists());
}

#[test]
fn fixtures_bool_track_compiles() {
    let f = fixtures::StoreFixture::with_bool_track("mask");
    assert!(f.path.exists());
}

#[test]
fn fixtures_single_col_uint32_compiles() {
    let f = fixtures::StoreFixture::with_single_col_uint32("depths");
    assert!(f.path.exists());
}
