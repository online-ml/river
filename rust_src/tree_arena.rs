use serde::{Deserialize, Serialize};

const LEAF_BIT: u32 = 1 << 31;
const INDEX_MASK: u32 = LEAF_BIT - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub(crate) struct NodeId(u32);

impl NodeId {
    #[inline]
    pub(crate) fn branch(index: usize) -> Self {
        let index = u32::try_from(index).expect("tree index exceeds u32");
        assert!(
            index <= INDEX_MASK,
            "tree index exceeds tagged node capacity"
        );
        Self(index)
    }

    #[inline]
    pub(crate) fn leaf(index: usize) -> Self {
        let index = u32::try_from(index).expect("tree index exceeds u32");
        assert!(
            index <= INDEX_MASK,
            "tree index exceeds tagged node capacity"
        );
        Self(index | LEAF_BIT)
    }

    #[inline]
    pub(crate) fn is_leaf(self) -> bool {
        self.0 & LEAF_BIT != 0
    }

    #[inline]
    pub(crate) fn index(self) -> usize {
        (self.0 & INDEX_MASK) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ids_preserve_kind_and_index() {
        let branch = NodeId::branch(17);
        let leaf = NodeId::leaf(17);

        assert!(!branch.is_leaf());
        assert!(leaf.is_leaf());
        assert_eq!(branch.index(), 17);
        assert_eq!(leaf.index(), 17);
        assert_ne!(branch, leaf);
    }
}
