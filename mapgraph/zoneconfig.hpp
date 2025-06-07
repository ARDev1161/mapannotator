#pragma once
/*-----------------------------------------------------------------------------
 *  zoneconfig.h
 *
 *  Enumerates all semantic zone classes used by the segmentation pipeline and
 *  provides helper predicates for quick category checks.
 *---------------------------------------------------------------------------*/
//#include <cstdint>
//#include <string>
#include "typeregistry.h"

struct ZoneFeatures
{
    double A;       ///< area  (m²)
    double P;       ///< perimeter (m)
    double C;       ///< compactness
    double AR;      ///< aspect ratio
    int    N;       ///< neighbours
    double w_avg;   ///< average passage width
    double w_min;   ///< minimal internal width
    // … add more if needed
};

namespace mapping {


///* -------- helper predicates ---------------------------------------------- */
//constexpr bool isPassage(ZoneType t) noexcept
//{
//    return t == ZoneType::Corridor ||
//           t == ZoneType::NarrowConnector ||
//           t == ZoneType::DoorArea;
//}
//constexpr bool isRoom(ZoneType t) noexcept
//{
//    return t == ZoneType::LivingRoomOfficeBedroom ||
//           t == ZoneType::StorageUtility ||
//           t == ZoneType::Sanitary ||
//           t == ZoneType::Kitchenette;
//}
//constexpr bool isOpenSpace(ZoneType t) noexcept
//{
//    return t == ZoneType::HallVestibule ||
//           t == ZoneType::AtriumLobby;
//}
//constexpr bool isTransitional(ZoneType t) noexcept
//{
//    return t == ZoneType::Staircase ||
//           t == ZoneType::ElevatorZone;
//}

} // namespace mapping
