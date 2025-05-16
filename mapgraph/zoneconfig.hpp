#pragma once
/*-----------------------------------------------------------------------------
 *  zoneconfig.h
 *
 *  Enumerates all semantic zone classes used by the segmentation pipeline and
 *  provides helper predicates for quick category checks.
 *---------------------------------------------------------------------------*/
namespace mapping {

enum class ZoneType : unsigned char
{
    /* Passage -------------------------------------------------------------- */
    Corridor,
    NarrowConnector,
    DoorArea,

    /* Room ----------------------------------------------------------------- */
    LivingRoomOfficeBedroom,
    StorageUtility,
    Sanitary,
    Kitchenette,

    /* Open space ----------------------------------------------------------- */
    HallVestibule,
    AtriumLobby,

    /* Transitional area ---------------------------------------------------- */
    Staircase,
    ElevatorZone,

    /* Fallback ------------------------------------------------------------- */
    Unknown
};

/* -------- helper predicates ---------------------------------------------- */
constexpr bool isPassage(ZoneType t) noexcept
{
    return t == ZoneType::Corridor ||
           t == ZoneType::NarrowConnector ||
           t == ZoneType::DoorArea;
}
constexpr bool isRoom(ZoneType t) noexcept
{
    return t == ZoneType::LivingRoomOfficeBedroom ||
           t == ZoneType::StorageUtility ||
           t == ZoneType::Sanitary ||
           t == ZoneType::Kitchenette;
}
constexpr bool isOpenSpace(ZoneType t) noexcept
{
    return t == ZoneType::HallVestibule ||
           t == ZoneType::AtriumLobby;
}
constexpr bool isTransitional(ZoneType t) noexcept
{
    return t == ZoneType::Staircase ||
           t == ZoneType::ElevatorZone;
}

} // namespace mapping
